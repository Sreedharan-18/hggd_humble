#!/usr/bin/env python3
import os
import sys
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Grasp, GripperTranslation
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def _resolve_hggd_root():
    env = os.environ.get("HGGD_ROOT", None)
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/workspace/HGGD/src/HGGD"))          # common mount
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "HGGD")                  # …/src/HGGD
    candidates.append(here.parents[1] / "HGGD")                  # …/pkg/HGGD

    for c in candidates:
        if c and c.exists() and (c / "dataset").is_dir() and (c / "models").is_dir():
            return str(c)
    return None

_hggd_guess = _resolve_hggd_root()
if _hggd_guess and _hggd_guess not in sys.path:
    sys.path.insert(0, _hggd_guess)

# ------------------------------------------------------------------------------
# HGGD imports (must succeed)
# ------------------------------------------------------------------------------
from dataset.config import get_camera_intrinsic
from dataset.evaluation import (
    anchor_output_process,
    collision_detect,
    detect_2d_grasp,
    detect_6d_grasp_multi,
)
from dataset.pc_dataset_tools import data_process, feature_fusion
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PointMultiGraspNet
from train_utils import *  # noqa: F401,F403 (for logging used by HGGD)

# ------------------------------------------------------------------------------
# Helpers (same math as in your file-based service)
# ------------------------------------------------------------------------------
def build_T_mats_from_graspgroup(gg):
    # Compose SE(3) from rotations & translations
    if hasattr(gg, 'rotation_matrices'):
        R = gg.rotation_matrices
    elif hasattr(gg, 'rotations'):
        R = gg.rotations
    else:
        raise AttributeError("GraspGroup missing rotations")
    if hasattr(gg, 'translations'):
        t = gg.translations
    elif hasattr(gg, 'translation'):
        t = gg.translation
    else:
        raise AttributeError("GraspGroup missing translations")

    R = np.asarray(R)
    t = np.asarray(t)
    N = R.shape[0]
    T = np.tile(np.eye(4, dtype=R.dtype)[None, ...], (N, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T

def rotmat_to_quat_xyzw(R):
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return float(qx), float(qy), float(qz), float(qw)

class PointCloudHelper:
    """
    Converts RGB-D to:
      - sampled point cloud with (xyz [+ rgb]) features
      - downsampled xyz maps for feature fusion
    Assumes 1280x720 intrinsics by default (HGGD realsense config).
    """
    def __init__(self, all_points_num) -> None:
        self.all_points_num = all_points_num
        self.output_shape = (80, 45)  # (W, H) for downsampled xyzs
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        # full-res maps (assumes 1280x720)
        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()
        # downscaled maps for xyz fusion
        ymap, xmap = np.meshgrid(np.arange(self.output_shape[1]),
                                 np.arange(self.output_shape[0]))
        factor = 1280 / self.output_shape[0]
        points_x = (xmap - cx / factor) / (fx / factor)
        points_y = (ymap - cy / factor) / (fy / factor)
        self.points_x_downscale = torch.from_numpy(points_x).float()
        self.points_y_downscale = torch.from_numpy(points_y).float()

    def to_scene_points(self, rgbs: torch.Tensor, depths: torch.Tensor, include_rgb=True):
        batch_size = rgbs.shape[0]
        feature_len = 3 + 3 * include_rgb
        points_all = -torch.ones(
            (batch_size, self.all_points_num, feature_len),
            dtype=torch.float32).cuda()
        masks = (depths > 0)
        cur_zs = depths / 1000.0
        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs
        for i in range(batch_size):
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], axis=-1)
            mask = masks[i]
            points = points[mask]
            colors = rgbs[i][:, mask].T
            if len(points) >= self.all_points_num:
                idxs = random.sample(range(len(points)), self.all_points_num)
                points = points[idxs]
                colors = colors[idxs]
            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points
        return points_all

    def to_xyz_maps(self, depths):
        downsample_depths = F.interpolate(depths[:, None],
                                          size=self.output_shape,
                                          mode='nearest').squeeze(1).cuda()
        cur_zs = downsample_depths / 1000.0
        cur_xs = self.points_x_downscale.cuda() * cur_zs
        cur_ys = self.points_y_downscale.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)

# ------------------------------------------------------------------------------
# ROS2 Node
# ------------------------------------------------------------------------------
class GraspPoseServicePerObject(Node):
    def __init__(self):
        super().__init__('grasp_pose_service_topics_per_object')
        self.bridge = CvBridge()
        self.last_rgb_np = None
        self.last_depth_np = None

        # ---------- Parameters ----------
        # Where HGGD lives (if you need to override)
        self.declare_parameter('hggd_root', '')
        # Camera topics
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        # Frame for published poses
        self.declare_parameter('frame_id', 'camera_color_optical_frame')

        # Model/data params (matching your CLI)
        self.declare_parameter('checkpoint_path', './realsense_checkpoint')
        self.declare_parameter('input_h', 360)
        self.declare_parameter('input_w', 640)
        self.declare_parameter('sigma', 10)
        self.declare_parameter('ratio', 8)
        self.declare_parameter('anchor_k', 6)
        self.declare_parameter('anchor_w', 50.0)
        self.declare_parameter('anchor_z', 20.0)
        self.declare_parameter('grid_size', 8)

        self.declare_parameter('anchor_num', 7)
        self.declare_parameter('all_points_num', 25600)
        self.declare_parameter('center_num', 48)
        self.declare_parameter('group_num', 512)

        self.declare_parameter('heatmap_thres', 0.01)
        self.declare_parameter('local_k', 10)
        self.declare_parameter('local_thres', 0.01)
        self.declare_parameter('rotation_num', 1)

        # Selection
        self.declare_parameter('topk_per_object', 2)

        # Publisher behavior
        self.declare_parameter('latched', True)
        self.declare_parameter('repeat_hz', 0.0)  # optional, republish last

        # Gripper mapping (width[m] -> joints)
        self.declare_parameter('gripper.max_width_m', 0.10)
        self.declare_parameter('gripper.open_pos', 0.00)
        self.declare_parameter('gripper.closed_pos', 0.80)
        self.declare_parameter('gripper.joint_names', ['gripper_left_joint', 'gripper_right_joint'])
        self.declare_parameter('approach_min_dist', 0.05)
        self.declare_parameter('approach_desired', 0.10)
        self.declare_parameter('retreat_min_dist', 0.05)
        self.declare_parameter('retreat_desired', 0.10)

        # Ensure HGGD path if provided
        hggd_root = self.get_parameter('hggd_root').get_parameter_value().string_value
        if hggd_root:
            if hggd_root not in sys.path:
                sys.path.insert(0, hggd_root)
            os.environ['HGGD_ROOT'] = hggd_root
            self.get_logger().info(f'Using hggd_root: {hggd_root}')
        elif _hggd_guess:
            self.get_logger().info(f'Using hggd_root guess: {_hggd_guess}')
        else:
            self.get_logger().warn('HGGD root not resolved—set param hggd_root or env HGGD_ROOT.')

        # ---------- Subscribers ----------
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value

        # Best-effort for camera topics is typical; use reliable if your driver supports
        sub_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(ROSImage, rgb_topic, self._rgb_cb, sub_qos)
        self.create_subscription(ROSImage, depth_topic, self._depth_cb, sub_qos)

        # ---------- Publishers ----------
        # One latched publisher per object ID (created on demand)
        self._obj_publishers: Dict[int, rclpy.publisher.Publisher] = {}

        # Optional combined stream (all objects)
        self._pub_all = self._make_publisher('top_grasps_all')

        # Cache last per-object msgs if you want to republish
        self._last_msgs: Dict[int, List[Grasp]] = {}

        # Optional periodic republish
        rep_hz = float(self.get_parameter('repeat_hz').get_parameter_value().double_value)
        self._rep_timer = None
        if rep_hz > 0.0:
            self._rep_timer = self.create_timer(1.0 / rep_hz, self._republish_last)

        # ---------- Models ----------
        self._init_models()
        self.get_logger().info('GraspPose service (per-object) ready. Call: ros2 service call /get_top_grasps std_srvs/srv/Trigger {}')

        # ---------- Service ----------
        self._srv = self.create_service(Trigger, 'get_top_grasps', self._handle_get_top_grasps)

    # --------------------------------------------------------------------------
    # Image callbacks
    # --------------------------------------------------------------------------
    def _rgb_cb(self, msg: ROSImage):
        # Convert to RGB8 numpy
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.last_rgb_np = np.asarray(rgb)
        except Exception as e:
            self.get_logger().warn(f'RGB conversion error: {e}')

    def _depth_cb(self, msg: ROSImage):
        # Convert to np (passthrough keeps original encoding)
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_np = np.asarray(depth)
            # Normalize to millimeters like your original file pipeline
            if depth_np.dtype == np.float32 or depth_np.dtype == np.float64:
                # depth is likely in meters → convert to mm
                depth_mm = (depth_np * 1000.0).astype(np.float32)
            else:
                # likely uint16 in mm already
                depth_mm = depth_np.astype(np.float32)
            # Clip to match original 0..1000mm constraint (adjust if needed)
            depth_mm = np.clip(depth_mm, 0, 1000)
            self.last_depth_np = depth_mm
        except Exception as e:
            self.get_logger().warn(f'Depth conversion error: {e}')

    # --------------------------------------------------------------------------
    # Model init
    # --------------------------------------------------------------------------
    def _init_models(self):
        P = lambda n: self.get_parameter(n).get_parameter_value()
        cp = P('checkpoint_path').string_value
        self.input_h = P('input_h').integer_value
        self.input_w = P('input_w').integer_value
        self.sigma = P('sigma').integer_value
        self.ratio = P('ratio').integer_value
        self.anchor_k = P('anchor_k').integer_value
        self.anchor_w = P('anchor_w').double_value
        self.anchor_z = P('anchor_z').double_value
        self.grid_size = P('grid_size').integer_value

        self.anchor_num = P('anchor_num').integer_value
        self.all_points_num = P('all_points_num').integer_value
        self.center_num = P('center_num').integer_value
        self.group_num = P('group_num').integer_value

        self.heatmap_thres = P('heatmap_thres').double_value
        self.local_k = P('local_k').integer_value
        self.local_thres = P('local_thres').double_value
        self.rotation_num = P('rotation_num').integer_value

        self.topk_per_object = P('topk_per_object').integer_value

        # Torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        np.set_printoptions(precision=4, suppress=True)
        torch.set_printoptions(precision=4, sci_mode=False)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        random.seed(123); np.random.seed(123); torch.manual_seed(123)

        # Networks
        self.anchornet = AnchorGraspNet(in_dim=4, ratio=self.ratio, anchor_k=self.anchor_k).cuda().eval()
        self.localnet = PointMultiGraspNet(info_size=3, k_cls=self.anchor_num**2).cuda().eval()

        # Load checkpoint & anchors
        ckpt = torch.load(cp)
        self.anchornet.load_state_dict(ckpt['anchor'])
        self.localnet.load_state_dict(ckpt['local'])
        basic_ranges = torch.linspace(-1, 1, self.anchor_num + 1).cuda()
        basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
        self.anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
        if 'gamma' in ckpt and 'beta' in ckpt:
            self.anchors['gamma'] = ckpt['gamma']
            self.anchors['beta'] = ckpt['beta']
            logging.info('Using saved anchors')
        print(f'-> loaded checkpoint {cp}')

        # Intrinsics
        self.K = get_camera_intrinsic()

    # --------------------------------------------------------------------------
    # Inference using latest frames
    # --------------------------------------------------------------------------
    def _run_inference_from_latest(self):
        if self.last_rgb_np is None or self.last_depth_np is None:
            return None

        # Convert to tensors like your file pipeline
        ori_rgb = (self.last_rgb_np / 255.0).astype(np.float32)
        ori_depth = self.last_depth_np.astype(np.float32)

        # transpose to (B,C,W,H) with your original (permute 2,1,0) convention
        ori_rgb_t = torch.from_numpy(ori_rgb).permute(2, 1, 0)[None].to('cuda', dtype=torch.float32)
        ori_depth_t = torch.from_numpy(ori_depth).T[None].to('cuda', dtype=torch.float32)

        # Pointcloud helper
        pc_helper = PointCloudHelper(all_points_num=self.all_points_num)
        view_points = pc_helper.to_scene_points(ori_rgb_t, ori_depth_t, include_rgb=True)
        # to_scene_points returns only points_all here
        points_all = view_points
        xyzs = pc_helper.to_xyz_maps(ori_depth_t)

        # Preprocess (note: original uses (input_w, input_h))
        rgb = F.interpolate(ori_rgb_t, (self.input_w, self.input_h))
        depth = F.interpolate(ori_depth_t[None], (self.input_w, self.input_h))[0] / 1000.0
        depth = torch.clip((depth - depth.mean()), -1, 1)
        x = torch.concat([depth[None], rgb], 1).to('cuda', dtype=torch.float32)

        with torch.no_grad():
            # 2D stage
            pred_2d, perpoint_features = self.anchornet(x)
            loc_map, cls_mask, theta_offset, height_offset, width_offset = \
                anchor_output_process(*pred_2d, sigma=self.sigma)

            rect_gg = detect_2d_grasp(
                loc_map, cls_mask, theta_offset, height_offset, width_offset,
                ratio=self.ratio, anchor_k=self.anchor_k,
                anchor_w=self.anchor_w, anchor_z=self.anchor_z,
                mask_thre=self.heatmap_thres,
                center_num=self.center_num, grid_size=self.grid_size,
                grasp_nms=self.grid_size, reduce='max')

            if rect_gg.size == 0:
                self.get_logger().warn('No 2D grasps found')
                return None

            # Feature fusion (+ build local groups)
            points_all = feature_fusion(points_all[..., :3], perpoint_features, xyzs)
            rect_ggs = [rect_gg]
            pc_group, valid_local_centers = data_process(
                points_all, ori_depth_t, rect_ggs,
                self.center_num, self.group_num,
                (self.input_w, self.input_h),
                min_points=32, is_training=False)
            rect_gg = rect_ggs[0]
            points_all = points_all.squeeze()

            # conditioning info (theta, width, depth)
            grasp_info = np.zeros((0, 3), dtype=np.float32)
            g_thetas = rect_gg.thetas[None]
            g_ws = rect_gg.widths[None]
            g_ds = rect_gg.depths[None]
            cur_info = np.vstack([g_thetas, g_ws, g_ds])
            grasp_info = np.vstack([grasp_info, cur_info.T])
            grasp_info = torch.from_numpy(grasp_info).to(dtype=torch.float32, device='cuda')

            # 3D refinement
            _, pred, offset = self.localnet(pc_group, grasp_info)
            _, pred_rect_gg = detect_6d_grasp_multi(
                rect_gg, pred, offset, valid_local_centers,
                (self.input_w, self.input_h), self.anchors, k=self.local_k)

            # Collision + NMS
            pred_gg, _ = collision_detect(
                points_all, pred_rect_gg.to_6d_grasp_group(depth=0.02),
                mode='graspnet')
            pred_gg = pred_gg.nms()

            if len(pred_gg) == 0:
                self.get_logger().warn('No valid 6D grasps after collision/NMS')
                return None

            return pred_gg

    # --------------------------------------------------------------------------
    # Gripper / MoveIt conversion
    # --------------------------------------------------------------------------
    def _width_to_joint_positions(self, width_m: float):
        P = lambda n: self.get_parameter(n).get_parameter_value()
        max_w = P('gripper.max_width_m').double_value
        open_pos = P('gripper.open_pos').double_value
        closed_pos = P('gripper.closed_pos').double_value
        joint_names = [s for s in P('gripper.joint_names').string_array_value]

        r = float(np.clip(width_m / max_w, 0.0, 1.0))
        pos = closed_pos + r * (open_pos - closed_pos)
        if len(joint_names) <= 1:
            return joint_names, [pos]
        else:
            return joint_names, [pos] * len(joint_names)

    def _make_moveit_grasp(self, T_4x4: np.ndarray, width_m: float, score: float, idx: int) -> Grasp:
        P = lambda n: self.get_parameter(n).get_parameter_value()
        frame_id = P('frame_id').string_value

        gpose = PoseStamped()
        gpose.header.frame_id = frame_id
        gpose.header.stamp = self.get_clock().now().to_msg()

        R = T_4x4[:3, :3]; t = T_4x4[:3, 3]
        qx, qy, qz, qw = rotmat_to_quat_xyzw(R)
        gpose.pose.position.x = float(t[0])
        gpose.pose.position.y = float(t[1])
        gpose.pose.position.z = float(t[2])
        gpose.pose.orientation.x = float(qx)
        gpose.pose.orientation.y = float(qy)
        gpose.pose.orientation.z = float(qz)
        gpose.pose.orientation.w = float(qw)

        joint_names, open_positions = self._width_to_joint_positions(P('gripper.max_width_m').double_value)
        _, close_positions = self._width_to_joint_positions(0.0)

        pre = JointTrajectory()
        pre.joint_names = joint_names
        pt_open = JointTrajectoryPoint()
        pt_open.positions = open_positions
        pt_open.time_from_start = Duration(sec=0, nanosec=500_000_000)
        pre.points = [pt_open]

        grasp_post = JointTrajectory()
        grasp_post.joint_names = joint_names
        pt_close = JointTrajectoryPoint()
        pt_close.positions = close_positions
        pt_close.time_from_start = Duration(sec=0, nanosec=700_000_000)
        grasp_post.points = [pt_close]

        z_axis = R[:, 2]
        app = GripperTranslation()
        app.direction.header.frame_id = frame_id
        app.direction.vector.x = float(z_axis[0])
        app.direction.vector.y = float(z_axis[1])
        app.direction.vector.z = float(z_axis[2])
        app.min_distance = P('approach_min_dist').double_value
        app.desired_distance = P('approach_desired').double_value

        ret = GripperTranslation()
        ret.direction.header.frame_id = frame_id
        ret.direction.vector.x = float(-z_axis[0])
        ret.direction.vector.y = float(-z_axis[1])
        ret.direction.vector.z = float(-z_axis[2])
        ret.min_distance = P('retreat_min_dist').double_value
        ret.desired_distance = P('retreat_desired').double_value

        msg = Grasp()
        msg.id = f'grasp_{idx}'
        msg.grasp_pose = gpose
        msg.grasp_quality = float(score)
        msg.pre_grasp_posture = pre
        msg.grasp_posture = grasp_post
        msg.pre_grasp_approach = app
        msg.post_grasp_retreat = ret
        return msg

    # --------------------------------------------------------------------------
    # Service: compute & publish top-K per object
    # --------------------------------------------------------------------------
    def _handle_get_top_grasps(self, request, response):
        pred_gg = self._run_inference_from_latest()
        if pred_gg is None:
            response.success = False
            response.message = 'No grasps found (frames missing or detection failed).'
            return response

        # Extract arrays
        T_all = build_T_mats_from_graspgroup(pred_gg)
        widths = np.asarray(pred_gg.widths) if hasattr(pred_gg, 'widths') else np.asarray(pred_gg.grasp_widths)
        scores = np.asarray(pred_gg.scores) if hasattr(pred_gg, 'scores') else (
            np.asarray(pred_gg.confidences) if hasattr(pred_gg, 'confidences') else np.asarray(pred_gg.confidence)
        )
        if hasattr(pred_gg, 'object_ids') and pred_gg.object_ids is not None:
            obj_ids = np.asarray(pred_gg.object_ids)
        else:
            # if not provided, assign everything to object 0
            obj_ids = np.zeros(len(scores), dtype=np.int32)

        unique_ids = np.unique(obj_ids)
        k = self.topk_per_object

        # Reset last cache
        self._last_msgs.clear()

        total_published = 0
        for oid in unique_ids:
            sel = np.where(obj_ids == oid)[0]
            if sel.size == 0:
                continue
            order = sel[np.argsort(-scores[sel])]
            keep = order[:min(k, len(order))]

            self.get_logger().info(f'Object {int(oid)}: publishing top-{len(keep)} grasps')
            pub = self._get_obj_publisher(int(oid))  # latched per object
            msgs_for_obj: List[Grasp] = []

            for rank, idx in enumerate(keep, start=1):
                T = T_all[idx]
                w = float(widths[idx])
                s = float(scores[idx])
                t = T[:3, 3]
                self.get_logger().info(
                    f'  #{rank}: score={s:.4f} center=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] width={w:.4f} (object_id={int(oid)})'
                )
                msg = self._make_moveit_grasp(T, w, s, idx)
                # Make ID include object id for clarity
                msg.id = f'obj{int(oid)}_grasp_{rank-1}'
                pub.publish(msg)
                self._pub_all.publish(msg)
                msgs_for_obj.append(msg)
                total_published += 1

            self._last_msgs[int(oid)] = msgs_for_obj

        response.success = total_published > 0
        response.message = f'Published {total_published} grasps across {len(unique_ids)} objects.'
        return response

    # --------------------------------------------------------------------------
    # QoS / Publishers
    # --------------------------------------------------------------------------
    def _make_publisher(self, name: str):
        latched = self.get_parameter('latched').get_parameter_value().bool_value
        if latched:
            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # latch latest
                depth=10,
                history=QoSHistoryPolicy.KEEP_LAST,
            )
        else:
            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=10,
                history=QoSHistoryPolicy.KEEP_LAST,
            )
        return self.create_publisher(Grasp, name, qos)

    def _get_obj_publisher(self, oid: int):
        if oid not in self._obj_publishers:
            topic = f'top_grasps/object_{oid}'
            self._obj_publishers[oid] = self._make_publisher(topic)
        return self._obj_publishers[oid]

    def _republish_last(self):
        # Optional periodic re-publish so late subscribers get the latest
        for oid, msgs in self._last_msgs.items():
            pub = self._get_obj_publisher(oid)
            for m in msgs:
                pub.publish(m)
                self._pub_all.publish(m)


def main():
    rclpy.init()
    node = GraspPoseServicePerObject()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
