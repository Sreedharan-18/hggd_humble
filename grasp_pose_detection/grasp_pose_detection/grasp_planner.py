#!/usr/bin/env python3
"""
RGB-D -> HGGD one-shot runner (no services, no MoveIt).
- Subscribes to RGB, depth, CameraInfo once (via cv_bridge)
- Runs HGGD on the captured pair (same pipeline as demo/service)
- Prints JSON with top-K grasps to stdout
- Publishes RViz axes markers on /hggd_grasps_markers
- Exits

Usage (example):
  ros2 run grasp_pose_detection rgbd_hggd_one_shot \
    --ros-args \
      -p hggd_root:=/ros2_ws/src/hggd_ros2/HGGD \
      -p checkpoint_path:=/ros2_ws/src/hggd_ros2/HGGD/realsense_checkpoint \
      -p rgb_topic:=/wrist_mounted_camera/image \
      -p depth_topic:=/wrist_mounted_camera/depth_image \
      -p camera_info_topic:=/wrist_mounted_camera/camera_info \
      -p input_h:=384 -p input_w:=640
"""

from __future__ import annotations
import os, sys, json, math, time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image as ROSImage, CameraInfo
from cv_bridge import CvBridge

# ─────────────────── HGGD path resolution ───────────────────
def _resolve_hggd_root(env_name="HGGD_ROOT") -> Optional[str]:
    env = os.environ.get(env_name)
    here = Path(__file__).resolve()
    cands = [env] if env else []
    cands += [
        here.parents[2] / "HGGD",             # …/src/HGGD
        here.parents[1] / "HGGD",
        "/ros2_ws/src/hggd_ros2/HGGD",
        "/workspace/HGGD/src/HGGD",
    ]
    for p in cands:
        p = Path(p) if p else None
        if p and (p / "dataset").is_dir() and (p / "models").is_dir():
            return str(p)
    return None

_hggd = _resolve_hggd_root()
if _hggd and _hggd not in sys.path:
    sys.path.insert(0, _hggd)

# ─────────────────── HGGD imports ───────────────────
from dataset.config import get_camera_intrinsic
from dataset.evaluation import (
    anchor_output_process, detect_2d_grasp, detect_6d_grasp_multi, collision_detect
)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet

# ─────────────────── helpers ───────────────────
def _pad_even_every_level(x: torch.Tensor, depth: int = 5):
    """Pad bottom/right so all encoder intermediate sizes remain EVEN."""
    _, _, H, W = x.shape

    def need_pad(dim: int) -> int:
        pad = 0
        while True:
            d = dim + pad
            ok = True
            for _ in range(depth - 1):
                d = (d - 4) // 2
                if d % 2:
                    ok = False
                    break
            if ok:
                return pad
            pad += 1

    ph, pw = need_pad(H), need_pad(W)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    return x, ph, pw

def build_T_mats_from_graspgroup(gg):
    R = np.asarray(getattr(gg, "rotation_matrices", getattr(gg, "rotations")))
    t = np.asarray(getattr(gg, "translations", getattr(gg, "translation")))
    T = np.tile(np.eye(4, dtype=R.dtype)[None], (R.shape[0], 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T

def rotmat_to_quat_xyzw(R):
    R = np.asarray(R, dtype=np.float64); tr = np.trace(R)
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx, qy, qz = (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            qw = (R[2,1]-R[1,2])/s; qx = 0.25*s
            qy, qz = (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
        elif R[1,1] > R[2,2]:
            s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            qw = (R[0,2]-R[2,0])/s; qy = 0.25*s
            qx, qz = (R[0,1]+R[1,0])/s, (R[1,2]+R[2,1])/s
        else:
            s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            qw = (R[1,0]-R[0,1])/s; qz = 0.25*s
            qx, qy = (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s
    return float(qx), float(qy), float(qz), float(qw)

class DynamicPointCloudHelper:
    """Build sampled XYZRGB and downsampled xyz maps from RGB-D + intrinsics."""
    def __init__(self, N: int):
        self.N = N
        self.points_x = self.points_y = None
        self.points_xd = self.points_yd = None
        self.out_shape = (45, 80)  # (H,W)

    def set_intrinsics(self, K: np.ndarray, w: int, h: int):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        self.points_x = torch.from_numpy((xs - cx)/fx).float()
        self.points_y = torch.from_numpy((ys - cy)/fy).float()
        Hd, Wd = self.out_shape
        ys_d, xs_d = np.meshgrid(np.arange(Hd), np.arange(Wd), indexing="ij")
        fx_d, fy_d, cx_d, cy_d = fx/w*Wd, fy/h*Hd, cx/w*Wd, cy/h*Hd
        self.points_xd = torch.from_numpy((xs_d - cx_d)/fx_d).float()
        self.points_yd = torch.from_numpy((ys_d - cy_d)/fy_d).float()

    def ready(self): return self.points_x is not None

    def to_scene_points(self, rgb: torch.Tensor, d_mm: torch.Tensor):
        B = rgb.shape[0]
        out = -torch.ones((B, self.N, 6), device=rgb.device)
        z = d_mm/1000.0; mask = d_mm > 0
        x = self.points_x.to(z.device)*z; y = self.points_y.to(z.device)*z
        for i in range(B):
            pts = torch.stack([x[i],y[i],z[i]], -1)[mask[i]]
            cols = rgb[i].permute(1,2,0).reshape(-1,3)[mask[i].reshape(-1)]
            if len(pts) >= self.N:
                idx = torch.randperm(len(pts), device=pts.device)[:self.N]
                pts, cols = pts[idx], cols[idx]
            out[i, :len(pts)] = torch.cat([pts, cols], 1)
        return out

    def to_xyz_maps(self, d_mm: torch.Tensor):
        Hd, Wd = self.out_shape
        z = F.interpolate(d_mm[:,None], (Hd,Wd), mode="nearest").squeeze(1)/1000.0
        x = self.points_xd.to(z.device)*z; y = self.points_yd.to(z.device)*z
        return torch.stack([x,y,z], 1)

# ─────────────────── Node ───────────────────
class OneShotHGGD(Node):
    def __init__(self):
        super().__init__("rgbd_hggd_one_shot")
        self.bridge = CvBridge()
        self.rgb = None      # np.ndarray HxWx3 (uint8)
        self.dmm = None      # np.ndarray HxW (float32 mm)
        self.K = get_camera_intrinsic()
        self.rgb_size = None
        self.cam_frame = None
        self.got_all = False

        # Params
        self.declare_parameter("hggd_root", "")
        self.declare_parameter("checkpoint_path", "./realsense_checkpoint")
        self.declare_parameter("rgb_topic", "/wrist_mounted_camera/image")
        self.declare_parameter("depth_topic", "/wrist_mounted_camera/depth_image")
        self.declare_parameter("camera_info_topic", "/wrist_mounted_camera/camera_info")
        self.declare_parameter("frame_id", "")
        self.declare_parameter("input_h", 384)
        self.declare_parameter("input_w", 640)
        self.declare_parameter("topk", 10)
        self.declare_parameter("viz_enable", True)
        self.declare_parameter("viz_axis_len", 0.08)
        self.declare_parameter("viz_axis_diam", 0.01)
        # HGGD hyper-params needed
        for n, v in [("sigma",10),("ratio",8),("anchor_k",6),("anchor_w",50.0),("anchor_z",20.0),
                     ("grid_size",8),("anchor_num",7),("all_points_num",25600),("center_num",48),
                     ("group_num",512),("heatmap_thres",0.01),("local_k",10),("local_thres",0.01),
                     ("rotation_num",1)]:
            self.declare_parameter(n, v)

        # Make HGGD root effective if user provided it here
        hroot = self.get_parameter("hggd_root").get_parameter_value().string_value
        if hroot and hroot not in sys.path:
            sys.path.insert(0, hroot); os.environ["HGGD_ROOT"] = hroot
        self.get_logger().info(f"HGGD_ROOT = {os.environ.get('HGGD_ROOT','?')}")

        # QoS matching camera streams
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1, history=QoSHistoryPolicy.KEEP_LAST
        )

        # Subscriptions
        self.create_subscription(ROSImage, self.get_parameter("rgb_topic").value, self._cb_rgb, qos)
        self.create_subscription(ROSImage, self.get_parameter("depth_topic").value, self._cb_d, qos)
        self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value, self._cb_ci, qos)

        # Markers
        self.viz_pub = self.create_publisher(MarkerArray, "hggd_grasps_markers", 10)
        self.marker_id = 0

        # Load models
        self._load_models()

        # Timer to check readiness
        self.timer = self.create_timer(0.1, self._maybe_run_once)

    # Callbacks
    def _cb_ci(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float32).reshape(3,3)
        self.cam_frame = msg.header.frame_id

    def _cb_rgb(self, msg: ROSImage):
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.rgb = np.asarray(img)
        self.rgb_size = (img.shape[1], img.shape[0])

    def _cb_d(self, msg: ROSImage):
        d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        if d.dtype in (np.float32, np.float64):
            self.dmm = (d.astype(np.float32) * 1000.0)  # meters->mm
        else:
            self.dmm = d.astype(np.float32)

    # Models
    def _load_models(self):
        P = lambda n: self.get_parameter(n).value
        ckpt = torch.load(P("checkpoint_path"), map_location="cuda")
        self.anchornet = AnchorGraspNet(in_dim=4, ratio=P("ratio"), anchor_k=P("anchor_k")).cuda().eval()
        self.localnet  = PointMultiGraspNet(info_size=3, k_cls=P("anchor_num")**2).cuda().eval()
        self.anchornet.load_state_dict(ckpt["anchor"])
        self.localnet.load_state_dict(ckpt["local"])
        basic = torch.linspace(-1,1,P("anchor_num")+1).cuda()
        self.anchors = {"gamma":(basic[1:]+basic[:-1])/2, "beta":(basic[1:]+basic[:-1])/2}
        if "gamma" in ckpt: self.anchors["gamma"] = ckpt["gamma"].to("cuda")
        if "beta"  in ckpt: self.anchors["beta"]  = ckpt["beta"].to("cuda")
        self.pc = DynamicPointCloudHelper(P("all_points_num"))

    # One-shot driver
    def _maybe_run_once(self):
        if self.got_all:  # already ran
            return
        if self.rgb is None or self.dmm is None:
            return
        if self.K is None or self.rgb_size is None:
            return

        # lock & run
        self.got_all = True
        try:
            res = self._run_hggd_once()
            if res is None:
                print(json.dumps({"success": False, "message": "no_grasps"}))
            else:
                print(json.dumps({"success": True, "grasps": res}, indent=2))
        except Exception as e:
            self.get_logger().error(f"one-shot failed: {e}")
            print(json.dumps({"success": False, "message": f"{type(e).__name__}: {e}"}))
        finally:
            # give RViz a moment to receive markers, then exit
            rclpy.task.Future()
            self.destroy_timer(self.timer)
            # shutdown from within the node
            self.get_logger().info("Done. Shutting down.")
            rclpy.shutdown()

    # Core inference
    def _run_hggd_once(self):
        P = lambda n: self.get_parameter(n).value
        h, w = int(P("input_h")), int(P("input_w"))

        # Intrinsics tied to the *actual* frame size
        self.pc.set_intrinsics(self.K, self.rgb_size[0], self.rgb_size[1])

        # tensors
        rgb_t = torch.from_numpy((self.rgb/255.0).astype(np.float32)).permute(2,0,1)[None].cuda()
        dmm_t = torch.from_numpy(self.dmm)[None].cuda()

        pts_all = self.pc.to_scene_points(rgb_t, dmm_t)
        xyzs    = self.pc.to_xyz_maps(dmm_t)

        # resize inputs for 2D stage
        rgb_rs = F.interpolate(rgb_t, (h, w))
        d_norm = F.interpolate((dmm_t/1000.0)[:,None], (h, w)).squeeze(1)
        d_norm = torch.clip(d_norm - d_norm.mean(), -1, 1)
        x = torch.cat([d_norm[:,None], rgb_rs], 1).to(dtype=torch.float32).cuda()

        # robust pad (fix skip connections)
        x, ph, pw = _pad_even_every_level(x, depth=5)

        with torch.no_grad():
            p2d, feat = self.anchornet(x)
            if ph or pw:
                p2d = [t[..., : -ph or None, : -pw or None] for t in p2d]

            loc_map, cls_mask, th_off, h_off, w_off = anchor_output_process(*p2d, sigma=P("sigma"))
            rect = detect_2d_grasp(
                loc_map, cls_mask, th_off, h_off, w_off,
                ratio=P("ratio"), anchor_k=P("anchor_k"),
                anchor_w=P("anchor_w"), anchor_z=P("anchor_z"),
                mask_thre=P("heatmap_thres"),
                center_num=P("center_num"), grid_size=P("grid_size"),
                grasp_nms=P("grid_size"), reduce="max"
            )
            if rect.size == 0:
                self.get_logger().warn("No 2D grasps found.")
                return None

            pts_all = feature_fusion(pts_all[...,:3], feat, xyzs)
            pc_group, _ = data_process(
                pts_all, dmm_t, [rect], P("center_num"), P("group_num"),
                (w, h), min_points=32, is_training=False
            )
            gi = torch.from_numpy(np.vstack([rect.thetas, rect.widths, rect.depths]).T.astype(np.float32)).cuda()
            _, pred, off = self.localnet(pc_group, gi)
            _, rect6d = detect_6d_grasp_multi(rect, pred, off, None, (w, h), self.anchors, k=P("local_k"))
            gg, _ = collision_detect(pts_all.squeeze(), rect6d.to_6d_grasp_group(depth=0.02), mode="graspnet")
            gg = gg.nms()
            if len(gg) == 0:
                self.get_logger().warn("No valid 6D grasps after collision/NMS.")
                return None

        # collect top-K (per object id)
        T_all = build_T_mats_from_graspgroup(gg)
        widths = np.asarray(getattr(gg,"widths",getattr(gg,"grasp_widths",np.full(len(gg),0.04))))
        scores = np.asarray(getattr(gg,"scores", getattr(gg,"confidence", np.zeros(len(gg)))))
        obj_ids = np.asarray(getattr(gg,"object_ids", np.zeros(len(gg), dtype=np.int32)))
        topk = int(P("topk"))

        out = []
        for oid in np.unique(obj_ids):
            sel = np.where(obj_ids == oid)[0]
            keep = sel[np.argsort(-scores[sel])][:min(topk, len(sel))]
            for idx in keep:
                T = T_all[idx]; R = T[:3,:3]; t = T[:3,3]
                qx,qy,qz,qw = rotmat_to_quat_xyzw(R)
                out.append(dict(
                    object_id=int(oid),
                    score=float(scores[idx]),
                    width_m=float(widths[idx]),
                    position_xyz=[float(v) for v in t],
                    orientation_xyzw=[qx,qy,qz,qw]
                ))

        # RViz markers
        self._publish_markers(T_all, count=len(out))
        return out

    def _publish_markers(self, T_all: np.ndarray, count: int):
        if not self.get_parameter("viz_enable").value or count == 0:
            return
        L = float(self.get_parameter("viz_axis_len").value)
        D = float(self.get_parameter("viz_axis_diam").value)
        frame = self.get_parameter("frame_id").value or self.cam_frame or "camera_color_optical_frame"
        colors = [(1,0,0,1),(0,1,0,1),(0,0,1,1)]
        ma = MarkerArray(); now = self.get_clock().now().to_msg()
        for i in range(count):
            T = T_all[i]; R, t = T[:3,:3], T[:3,3]
            origin = Point(x=float(t[0]), y=float(t[1]), z=float(t[2]))
            for a, ax in enumerate(R.T):
                tip = Point(x=float(t[0]+L*ax[0]), y=float(t[1]+L*ax[1]), z=float(t[2]+L*ax[2]))
                m = Marker()
                m.header = Header(frame_id=frame, stamp=now)
                m.ns = "grasp_axes"; m.id = self.marker_id
                m.type = Marker.ARROW; m.action = Marker.ADD
                m.lifetime = Duration(sec=0, nanosec=800_000_000)
                m.scale = Vector3(x=D, y=D*2, z=D*2)
                m.points = [origin, tip]
                r,g,b,a = colors[a]
                m.color.r, m.color.g, m.color.b, m.color.a = r,g,b,a
                ma.markers.append(m); self.marker_id += 1
        self.viz_pub.publish(ma)

# ─────────────────── main ───────────────────
def main():
    rclpy.init()
    node = OneShotHGGD()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # if we got here without node->shutdown, do it now
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
