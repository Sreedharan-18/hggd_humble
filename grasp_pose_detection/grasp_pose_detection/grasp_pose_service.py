#!/usr/bin/env python3
# Grasp-Pose online service using HGGD – ROS 2 (no MoveIt dependency).
# Publishes MarkerArray and returns top-K grasps as JSON in Trigger response.

from __future__ import annotations
import os, sys, json, math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch, torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image as ROSImage, CameraInfo
from cv_bridge import CvBridge
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray

# ────────────────────────── find + add HGGD root ────────────────────────────
def _resolve_hggd() -> Optional[str]:
    env = os.environ.get("HGGD_ROOT")
    here = Path(__file__).resolve()
    cands = [env] if env else []
    cands += [here.parents[2]/"HGGD", here.parents[1]/"HGGD",
              "/ros2_ws/src/hggd_ros2/HGGD", "/workspace/HGGD/src/HGGD"]
    for p in cands:
        p = Path(p) if p else None
        if p and (p/"dataset").is_dir() and (p/"models").is_dir():
            return str(p)
    return None

_hggd = _resolve_hggd()
if _hggd and _hggd not in sys.path:
    sys.path.insert(0, _hggd)

# ────────────────────────── HGGD imports ─────────────────────────────────────
from dataset.config import get_camera_intrinsic
from dataset.evaluation import (anchor_output_process, detect_2d_grasp,
                                detect_6d_grasp_multi, collision_detect)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet

# ────────────────────────── robust padding helper ───────────────────────────
def _pad_even_every_level(x: torch.Tensor, depth: int = 5) -> Tuple[torch.Tensor,int,int]:
    """
    Pads (B,C,H,W) on bottom/right so that after *depth* encoder halvings using
      d_next = floor((d - 4) / 2)
    all intermediate d are EVEN. This matches AnchorGraspNet skip structure.

    Returns (padded_x, pad_h, pad_w).
    """
    _,_,H,W = x.shape

    def need_pad(dim: int) -> int:
        pad = 0
        while True:
            d = dim + pad
            ok = True
            # We require even for every level *before* the last output fuse
            for _ in range(depth - 1):
                d = (d - 4) // 2
                if d % 2:  # odd → mismatch later
                    ok = False
                    break
            if ok:
                return pad
            pad += 1

    ph, pw = need_pad(H), need_pad(W)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))  # (L,R,T,B)
    return x, ph, pw

# ────────────────────────── math helpers ─────────────────────────────────────
def build_T_mats_from_graspgroup(gg):
    R = np.asarray(getattr(gg,"rotation_matrices", getattr(gg,"rotations")))
    t = np.asarray(getattr(gg,"translations", getattr(gg,"translation")))
    T = np.tile(np.eye(4,dtype=R.dtype)[None], (R.shape[0],1,1))
    T[:,:3,:3], T[:,:3,3] = R, t
    return T

def rotmat_to_quat_xyzw(R):
    R = np.asarray(R,dtype=np.float64); tr=np.trace(R)
    if tr>0:
        s=math.sqrt(tr+1.0)*2.0; qw=0.25*s
        qx=(R[2,1]-R[1,2])/s; qy=(R[0,2]-R[2,0])/s; qz=(R[1,0]-R[0,1])/s
    else:
        if R[0,0]>R[1,1] and R[0,0]>R[2,2]:
            s=math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2.0
            qw=(R[2,1]-R[1,2])/s; qx=0.25*s
            qy=(R[0,1]+R[1,0])/s; qz=(R[0,2]+R[2,0])/s
        elif R[1,1]>R[2,2]:
            s=math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2.0
            qw=(R[0,2]-R[2,0])/s; qy=0.25*s
            qx=(R[0,1]+R[1,0])/s; qz=(R[1,2]+R[2,1])/s
        else:
            s=math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2.0
            qw=(R[1,0]-R[0,1])/s; qz=0.25*s
            qx=(R[0,2]+R[2,0])/s; qy=(R[1,2]+R[2,1])/s
    return float(qx),float(qy),float(qz),float(qw)

# ────────────────────────── point-cloud helper ───────────────────────────────
class DynamicPointCloudHelper:
    def __init__(self,N:int):
        self.N=N
        self.points_x=self.points_y=None
        self.points_xd=self.points_yd=None
        self.out_shape=(45,80)

    def set_intrinsics(self,K:np.ndarray,w:int,h:int):
        fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
        ys,xs=np.meshgrid(np.arange(h),np.arange(w),indexing="ij")
        self.points_x=torch.from_numpy((xs-cx)/fx).float()
        self.points_y=torch.from_numpy((ys-cy)/fy).float()
        Hd,Wd=self.out_shape
        ys_d,xs_d=np.meshgrid(np.arange(Hd),np.arange(Wd),indexing="ij")
        fx_d,fy_d,cx_d,cy_d=fx/w*Wd,fy/h*Hd,cx/w*Wd,cy/h*Hd
        self.points_xd=torch.from_numpy((xs_d-cx_d)/fx_d).float()
        self.points_yd=torch.from_numpy((ys_d-cy_d)/fy_d).float()

    def ready(self): return self.points_x is not None

    def to_scene_points(self,rgb:torch.Tensor,depth:torch.Tensor):
        B=rgb.shape[0]; out=-torch.ones((B,self.N,6),device=rgb.device)
        z=depth/1000.0; mask=depth>0
        x=self.points_x.to(z.device)*z; y=self.points_y.to(z.device)*z
        for i in range(B):
            pts=torch.stack([x[i],y[i],z[i]],-1)[mask[i]]
            cols=rgb[i].permute(1,2,0).reshape(-1,3)[mask[i].reshape(-1)]
            if len(pts)>=self.N:
                idx=torch.randperm(len(pts),device=pts.device)[:self.N]
                pts,cols=pts[idx],cols[idx]
            out[i,:len(pts)]=torch.cat([pts,cols],1)
        return out

    def to_xyz_maps(self,d_mm:torch.Tensor):
        Hd,Wd=self.out_shape
        z=F.interpolate(d_mm[:,None],(Hd,Wd),mode="nearest").squeeze(1)/1000.0
        x=self.points_xd.to(z.device)*z; y=self.points_yd.to(z.device)*z
        return torch.stack([x,y,z],1)

# ────────────────────────── ROS 2 node ───────────────────────────────────────
class GraspPoseService(Node):
    def __init__(self):
        super().__init__("hggd_service")
        self.bridge=CvBridge()
        self.last_rgb=self.last_dmm=None
        self.rgb_size=None

        # params
        self.declare_parameter("hggd_root","")
        self.declare_parameter("rgb_topic","/wrist_mounted_camera/image")
        self.declare_parameter("depth_topic","/wrist_mounted_camera/depth_image")
        self.declare_parameter("camera_info_topic","/wrist_mounted_camera/camera_info")
        self.declare_parameter("frame_id","")
        self.declare_parameter("checkpoint_path","./realsense_checkpoint")
        self.declare_parameter("input_h",360)
        self.declare_parameter("input_w",640)
        self.declare_parameter("topk_per_object",10)
        self.declare_parameter("viz_enable",True)
        self.declare_parameter("viz_axis_len",0.08)
        self.declare_parameter("viz_axis_diam",0.01)
        for n,v in [("sigma",10),("ratio",8),("anchor_k",6),("anchor_w",50.0),("anchor_z",20.0),
                    ("grid_size",8),("anchor_num",7),("all_points_num",25600),("center_num",48),
                    ("group_num",512),("heatmap_thres",0.01),("local_k",10),("local_thres",0.01),
                    ("rotation_num",1)]: self.declare_parameter(n,v)

        root_p=self.get_parameter("hggd_root").get_parameter_value().string_value
        if root_p and root_p not in sys.path:
            sys.path.insert(0,root_p)
            os.environ["HGGD_ROOT"]=root_p
        self.get_logger().info(f"HGGD_ROOT = {os.environ.get('HGGD_ROOT','?')}")

        self.K=get_camera_intrinsic()
        self.cam_frame=None
        self.pc=DynamicPointCloudHelper(self.get_parameter("all_points_num").value)

        qos=QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                       durability=QoSDurabilityPolicy.VOLATILE,
                       depth=1, history=QoSHistoryPolicy.KEEP_LAST)
        self.create_subscription(ROSImage,self.get_parameter("rgb_topic").value,self._cb_rgb,qos)
        self.create_subscription(ROSImage,self.get_parameter("depth_topic").value,self._cb_d,qos)
        self.create_subscription(CameraInfo,self.get_parameter("camera_info_topic").value,self._cb_ci,qos)

        self.viz_pub=self.create_publisher(MarkerArray,"hggd_grasps_markers",10)
        self.marker_id=0

        self._load_models()

        self.create_service(Trigger,"get_top_grasps",self._srv_cb)
        self.get_logger().info("🚀  HGGD service ready — call: ros2 service call /get_top_grasps std_srvs/srv/Trigger {}")

    # callbacks
    def _cb_ci(self,msg:CameraInfo):
        self.K=np.array(msg.k,dtype=np.float32).reshape(3,3)
        self.cam_frame=msg.header.frame_id
        self.pc.set_intrinsics(self.K,msg.width,msg.height)

    def _cb_rgb(self,msg:ROSImage):
        rgb=self.bridge.imgmsg_to_cv2(msg,"rgb8")
        self.last_rgb=np.asarray(rgb)
        self.rgb_size=(rgb.shape[1],rgb.shape[0])
        if self.K is not None and not self.pc.ready():
            self.pc.set_intrinsics(self.K,*self.rgb_size)

    def _cb_d(self,msg:ROSImage):
        d=self.bridge.imgmsg_to_cv2(msg,"passthrough")
        self.last_dmm=(d*1000.0).astype(np.float32) if d.dtype==np.float32 else d.astype(np.float32)

    # nets
    def _load_models(self):
        P=lambda n:self.get_parameter(n).value
        ckpt=torch.load(P("checkpoint_path"),map_location="cuda")
        self.anchornet=AnchorGraspNet(in_dim=4,ratio=P("ratio"),anchor_k=P("anchor_k")).cuda().eval()
        self.localnet=PointMultiGraspNet(info_size=3,k_cls=P("anchor_num")**2).cuda().eval()
        self.anchornet.load_state_dict(ckpt["anchor"])
        self.localnet.load_state_dict(ckpt["local"])
        basic=torch.linspace(-1,1,P("anchor_num")+1).cuda()
        self.anchors={"gamma":(basic[1:]+basic[:-1])/2, "beta":(basic[1:]+basic[:-1])/2}
        if "gamma" in ckpt: self.anchors["gamma"]=ckpt["gamma"].to("cuda")
        if "beta" in ckpt: self.anchors["beta"]=ckpt["beta"].to("cuda")

    # inference
    def _infer(self):
        if self.last_rgb is None or self.last_dmm is None:
            self.get_logger().warning("Waiting for RGB-D frames…")
            return None
        if not self.pc.ready():
            self.pc.set_intrinsics(self.K,*self.rgb_size)

        rgb_t=torch.from_numpy((self.last_rgb/255.).astype(np.float32)).permute(2,0,1)[None].cuda()
        dmm_t=torch.from_numpy(self.last_dmm)[None].cuda()
        pts_all=self.pc.to_scene_points(rgb_t,dmm_t)
        xyzs=self.pc.to_xyz_maps(dmm_t)

        P=lambda n:self.get_parameter(n).value
        h,w=P("input_h"),P("input_w")
        rgb_rs=F.interpolate(rgb_t,(h,w))
        d_norm=F.interpolate((dmm_t/1000.)[:,None],(h,w)).squeeze(1)
        d_norm=torch.clip(d_norm-d_norm.mean(),-1,1)
        x=torch.cat([d_norm[:,None],rgb_rs],1).to(torch.float32, device="cuda")

        # critical: robust padding for AnchorGraspNet skip sizes
        x,ph,pw=_pad_even_every_level(x, depth=5)

        with torch.no_grad():
            p2d,feat=self.anchornet(x)
            if ph or pw:
                # crop back to original target size
                p2d=[t[..., : -ph or None, : -pw or None] for t in p2d]

            loc_map,cls_mask,th_off,h_off,w_off = anchor_output_process(*p2d, sigma=P("sigma"))
            rect = detect_2d_grasp(loc_map,cls_mask,th_off,h_off,w_off,
                                   ratio=P("ratio"),anchor_k=P("anchor_k"),
                                   anchor_w=P("anchor_w"),anchor_z=P("anchor_z"),
                                   mask_thre=P("heatmap_thres"),
                                   center_num=P("center_num"),grid_size=P("grid_size"),
                                   grasp_nms=P("grid_size"),reduce="max")
            if rect.size==0:
                return None

            pts_all=feature_fusion(pts_all[...,:3],feat,xyzs)
            pc_group,_=data_process(pts_all,dmm_t,[rect],
                                    P("center_num"),P("group_num"),
                                    (w,h),min_points=32,is_training=False)
            gi=torch.from_numpy(np.vstack([rect.thetas,rect.widths,rect.depths]).T.astype(np.float32)).cuda()
            _,pred,off=self.localnet(pc_group,gi)
            _,rect6d=detect_6d_grasp_multi(rect,pred,off,None,(w,h),self.anchors,k=P("local_k"))
            gg,_=collision_detect(pts_all.squeeze(),rect6d.to_6d_grasp_group(depth=0.02),mode="graspnet")
            return gg.nms()

    # markers
    def _publish_markers(self,Ts:List[np.ndarray]):
        if not Ts or not self.get_parameter("viz_enable").value: return
        L=self.get_parameter("viz_axis_len").value
        D=self.get_parameter("viz_axis_diam").value
        frame = self.get_parameter("frame_id").value or self.cam_frame or "camera_color_optical_frame"
        colors=[(1,0,0,1),(0,1,0,1),(0,0,1,1)]
        ma=MarkerArray(); now=self.get_clock().now().to_msg()
        for T in Ts:
            R,t=T[:3,:3],T[:3,3]
            origin=Point(x=float(t[0]),y=float(t[1]),z=float(t[2]))
            for a,ax in enumerate(R.T):
                tip=Point(x=float(t[0]+L*ax[0]),y=float(t[1]+L*ax[1]),z=float(t[2]+L*ax[2]))
                m=Marker()
                m.header=Header(frame_id=frame, stamp=now)
                m.ns="grasp_axes"; m.id=self.marker_id
                m.type=Marker.ARROW; m.action=Marker.ADD
                m.lifetime=Duration(sec=0,nanosec=800_000_000)
                m.scale=Vector3(x=D,y=D*2,z=D*2)
                m.points=[origin, tip]
                r,g,b,a=colors[a]
                m.color.r=r; m.color.g=g; m.color.b=b; m.color.a=a
                ma.markers.append(m); self.marker_id+=1
        self.viz_pub.publish(ma)

    # service
    def _srv_cb(self,_,resp):
        gg=self._infer()
        if gg is None:
            resp.success=False; resp.message="no_grasps"; return resp
        k=self.get_parameter("topk_per_object").value
        T_all=build_T_mats_from_graspgroup(gg)
        widths=np.asarray(getattr(gg,"widths",getattr(gg,"grasp_widths",np.full(len(gg),0.04))))
        scores=np.asarray(getattr(gg,"scores",getattr(gg,"confidence",np.zeros(len(gg)))))
        obj_ids=np.asarray(getattr(gg,"object_ids",np.zeros(len(gg),dtype=np.int32)))
        result=[]
        for oid in np.unique(obj_ids):
            sel=np.where(obj_ids==oid)[0]
            for idx in sel[np.argsort(-scores[sel])][:min(k,len(sel))]:
                T=T_all[idx]; qx,qy,qz,qw=rotmat_to_quat_xyzw(T[:3,:3]); t=T[:3,3]
                result.append(dict(object_id=int(oid),score=float(scores[idx]),width_m=float(widths[idx]),
                                   position_xyz=[float(p) for p in t],
                                   orientation_xyzw=[qx,qy,qz,qw]))
        self._publish_markers([T_all[i] for i in range(len(result))])
        resp.success=True; resp.message=json.dumps(result); return resp

def main():
    rclpy.init()
    node=GraspPoseService()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node(); rclpy.shutdown()

if __name__=="__main__":
    main()
