#!/usr/bin/env python3
import os
import sys
import random
from time import time
from pathlib import Path
import argparse  # not strictly needed, kept for parity

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# -----------------------------
# Resolve HGGD repo on sys.path
# -----------------------------
def _resolve_hggd_root():
    """
    Try several ways to find the HGGD root that contains:
      dataset/, models/, train_utils.py, images/, realsense_checkpoint, etc.
    Priority:
      1) ROS param 'hggd_root' (handled after rclpy init)
      2) Env var HGGD_ROOT
      3) Common absolute fallback: /workspace/HGGD/src/HGGD
      4) Relative to this file: ../../HGGD or ../HGGD
    """
    env = os.environ.get("HGGD_ROOT", None)
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/workspace/HGGD/src/HGGD"))
    here = Path(__file__).resolve()
    candidates.append(here.parents[2] / "HGGD")  # …/src/HGGD
    candidates.append(here.parents[1] / "HGGD")  # …/grasp_pose_detection/HGGD

    for c in candidates:
        if c is None:
            continue
        if c.exists() and (c / "dataset").is_dir() and (c / "models").is_dir():
            return str(c)
    return None

# Tentatively add a best-guess HGGD root before importing HGGD modules.
# (We may override this later if a ROS param is provided.)
_hggd_root_guess = _resolve_hggd_root()
if _hggd_root_guess and _hggd_root_guess not in sys.path:
    sys.path.insert(0, _hggd_root_guess)

# -----------------------------
# HGGD imports (now visible)
# -----------------------------
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
from train_utils import *  # noqa: F401,F403 (uses logging etc.)

# Open3D is optional here (no viz in service). Import if present.
try:
    import open3d as o3d  # noqa: F401
except Exception:
    o3d = None

# -----------------------------
# ROS 2 / MoveIt 2 imports
# -----------------------------
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.msg import Grasp, GripperTranslation


# -----------------------------
# Core helpers (same math as before)
# -----------------------------
class PointCloudHelper:
    """
    Converts RGB-D to:
      - sampled point cloud with (xyz [+ rgb]) features
      - downsampled xyz maps for feature fusion
    Assumes source images are 1280x720 (adjust if different).
    """
    def __init__(self, all_points_num) -> None:
        self.all_points_num = all_points_num
        self.output_shape = (80, 45)  # (W, H) for downsampled xyzs
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        # full-res maps (assumes 1280x720 source)
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
        idxs = []
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
                cur_idxs = random.sample(range(len(points)), self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                idxs.append(cur_idxs)
            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points
        return points_all, idxs, masks

    def to_xyz_maps(self, depths):
        downsample_depths = F.interpolate(depths[:, None],
                                          size=self.output_shape,
                                          mode='nearest').squeeze(1).cuda()
        cur_zs = downsample_depths / 1000.0
        cur_xs = self.points_x_downscale.cuda() * cur_zs
        cur_ys = self.points_y_downscale.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)


def build_T_mats_from_graspgroup(gg):
    """Compose SE(3) matrices from rotations and translations on a GraspGroup."""
    if hasattr(gg, 'rotation_matrices'):
        R = gg.rotation_matrices
    elif hasattr(gg, 'rotations'):
        R = gg.rotations
    else:
        raise AttributeError("GraspGroup has no rotation_matrices/rotations")
    if hasattr(gg, 'translations'):
        t = gg.translations
    elif hasattr(gg, 'translation'):
        t = gg.translation
    else:
        raise AttributeError("GraspGroup has no translations/translation")
    R = np.asarray(R)
    t = np.asarray(t)
    N = R.shape[0]
    T = np.tile(np.eye(4, dtype=R.dtype)[None, ...], (N, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def rotmat_to_quat_xyzw(R):
    """Convert a proper rotation matrix to quaternion (x,y,z,w)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        # find major diagonal element
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


# -----------------------------
# ROS2 node: service that publishes MoveIt2 Grasps
# -----------------------------
class GraspPoseService(Node):
    def __init__(self):
        super().__init__('grasp_pose_service')

        # Allow overriding HGGD path at runtime
        self.declare_parameter('hggd_root', '')

        # --------- Parameters (mirror your CLI, with sane defaults) ---------
        # model / data
        self.declare_parameter('checkpoint_path', './realsense_checkpoint')
        self.declare_parameter('rgb_path', './images/demo_rgb.png')
        self.declare_parameter('depth_path', './images/demo_depth.png')

        # network sizes / knobs
        self.declare_parameter('input_h', 360)
        self.declare_parameter('input_w', 640)
        self.declare_parameter('sigma', 10)
        self.declare_parameter('ratio', 8)
        self.declare_parameter('anchor_k', 6)
        self.declare_parameter('anchor_w', 50.0)
        self.declare_parameter('anchor_z', 20.0)
        self.declare_parameter('grid_size', 8)

        # point cloud sampling
        self.declare_parameter('anchor_num', 7)
        self.declare_parameter('all_points_num', 25600)
        self.declare_parameter('center_num', 48)
        self.declare_parameter('group_num', 512)

        # detection thresholds
        self.declare_parameter('heatmap_thres', 0.01)
        self.declare_parameter('local_k', 10)
        self.declare_parameter('local_thres', 0.01)
        self.declare_parameter('rotation_num', 1)

        # grasp selection
        self.declare_parameter('topk', 5)

        # frames & approach/retreat (MoveIt GripperTranslation)
        self.declare_parameter('frame_id', 'camera_color_optical_frame')
        self.declare_parameter('approach_min_dist', 0.05)   # meters
        self.declare_parameter('approach_desired', 0.10)    # meters
        self.declare_parameter('retreat_min_dist', 0.05)    # meters
        self.declare_parameter('retreat_desired', 0.10)     # meters

        # gripper mapping (width[m] -> joint positions)
        self.declare_parameter('gripper.max_width_m', 0.10)   # full-open jaw gap in meters
        self.declare_parameter('gripper.open_pos', 0.00)      # joint position at full open
        self.declare_parameter('gripper.closed_pos', 0.80)    # joint position at fully closed
        self.declare_parameter('gripper.joint_names', ['gripper_left_joint', 'gripper_right_joint'])

        # --------- Ensure HGGD is on sys.path (final chance) ----------
        hggd_root_param = self.get_parameter('hggd_root').get_parameter_value().string_value
        if hggd_root_param:
            if hggd_root_param not in sys.path:
                sys.path.insert(0, hggd_root_param)
            os.environ['HGGD_ROOT'] = hggd_root_param  # persist for child imports
            self.get_logger().info(f'Using hggd_root from param: {hggd_root_param}')
        else:
            if _hggd_root_guess:
                self.get_logger().info(f'Using hggd_root guess: {_hggd_root_guess}')
            else:
                self.get_logger().warn('Could not resolve HGGD root automatically. '
                                       'Set param hggd_root or env HGGD_ROOT.')

        # --------- Create publisher and service ----------
        self.grasp_pub = self.create_publisher(Grasp, 'top_grasps', 10)
        self.srv = self.create_service(Trigger, 'get_top_grasps', self.handle_get_top_grasps)

        # --------- Initialize models & static stuff once ----------
        self.get_logger().info('Initializing models...')
        self.init_models_and_data()

        self.get_logger().info('GraspPoseService ready. Call: ros2 service call /get_top_grasps std_srvs/srv/Trigger {}')

    # ----------------- model & data init -----------------
    def init_models_and_data(self):
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

        self.topk = P('topk').integer_value

        # Torch / seeds
        np.set_printoptions(precision=4, suppress=True)
        torch.set_printoptions(precision=4, sci_mode=False)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        random.seed(123); np.random.seed(123); torch.manual_seed(123)

        # Models
        self.anchornet = AnchorGraspNet(in_dim=4, ratio=self.ratio, anchor_k=self.anchor_k).cuda().eval()
        self.localnet = PointMultiGraspNet(info_size=3, k_cls=self.anchor_num**2).cuda().eval()

        # Load checkpoint & anchors
        check_point = torch.load(cp)
        self.anchornet.load_state_dict(check_point['anchor'])
        self.localnet.load_state_dict(check_point['local'])

        basic_ranges = torch.linspace(-1, 1, self.anchor_num + 1).cuda()
        basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
        self.anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
        if 'gamma' in check_point and 'beta' in check_point:
            self.anchors['gamma'] = check_point['gamma']
            self.anchors['beta'] = check_point['beta']
            logging.info('Using saved anchors')
        print(f'-> loaded checkpoint {cp}')

        # Intrinsics
        self.K = get_camera_intrinsic()

    # ----------------- core inference (no viz; returns top-k arrays) -----------------
    def run_inference_once(self, rgb_path: str, depth_path: str):
        # read inputs
        ori_depth = np.array(Image.open(depth_path))
        ori_rgb = np.array(Image.open(rgb_path)) / 255.0
        ori_depth = np.clip(ori_depth, 0, 1000)
        ori_rgb = torch.from_numpy(ori_rgb).permute(2, 1, 0)[None]
        ori_rgb = ori_rgb.to(device='cuda', dtype=torch.float32)
        ori_depth = torch.from_numpy(ori_depth).T[None]
        ori_depth = ori_depth.to(device='cuda', dtype=torch.float32)

        # pc helper + 3D structures
        pc_helper = PointCloudHelper(all_points_num=self.all_points_num)
        view_points, _, _ = pc_helper.to_scene_points(ori_rgb, ori_depth, include_rgb=True)
        xyzs = pc_helper.to_xyz_maps(ori_depth)

        # preprocess for net
        rgb = F.interpolate(ori_rgb, (self.input_w, self.input_h))  # note: (W,H) per your original
        depth = F.interpolate(ori_depth[None], (self.input_w, self.input_h))[0] / 1000.0
        depth = torch.clip((depth - depth.mean()), -1, 1)
        x = torch.concat([depth[None], rgb], 1).to(device='cuda', dtype=torch.float32)

        with torch.no_grad():
            # ---- 2D stage ----
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

            # ---- fusion & local 3D neighborhoods ----
            points_all = feature_fusion(view_points[..., :3], perpoint_features, xyzs)
            rect_ggs = [rect_gg]
            pc_group, valid_local_centers = data_process(
                points_all, ori_depth, rect_ggs,
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

            # ---- 3D refinement ----
            _, pred, offset = self.localnet(pc_group, grasp_info)
            _, pred_rect_gg = detect_6d_grasp_multi(
                rect_gg, pred, offset, valid_local_centers,
                (self.input_w, self.input_h), self.anchors, k=self.local_k)

            # ---- collision & NMS ----
            pred_gg, _ = collision_detect(
                points_all, pred_rect_gg.to_6d_grasp_group(depth=0.02),
                mode='graspnet')
            pred_gg = pred_gg.nms()

            if len(pred_gg) == 0:
                self.get_logger().warn('No valid 6D grasps after collision/NMS')
                return None

            # ---- TOP-K ----
            T_mats = build_T_mats_from_graspgroup(pred_gg)
            widths = np.asarray(pred_gg.widths) if hasattr(pred_gg, 'widths') else np.asarray(pred_gg.grasp_widths)
            scores = np.asarray(pred_gg.scores) if hasattr(pred_gg, 'scores') else (
                np.asarray(pred_gg.confidences) if hasattr(pred_gg, 'confidences') else np.asarray(pred_gg.confidence)
            )

            order = np.argsort(-scores)
            kmax = min(self.topk, len(order))
            keep = [int(i) for i in order[:kmax]]

            # Print once here (service call)
            self.get_logger().info(f'Top-{kmax} grasps (camera frame):')
            for rank, idx in enumerate(keep, start=1):
                t = T_mats[idx][:3, 3]
                w = widths[idx]
                s = scores[idx]
                self.get_logger().info(f'  #{rank}: score={s:.4f}  center_xyz=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m  width={w:.4f} m')

            return {
                "T_mats": T_mats[keep],
                "widths": widths[keep],
                "scores": scores[keep],
            }

    # ----------------- width -> joint mapping -----------------
    def width_to_joint_positions(self, width_m: float):
        """
        Map gripper 'width' (meters) to joint position(s).
        Linear mapping between (0..max_width_m) -> (closed_pos..open_pos).
        If two joint names are provided, mirror the same value to both.
        """
        P = lambda n: self.get_parameter(n).get_parameter_value()
        max_w = P('gripper.max_width_m').double_value
        open_pos = P('gripper.open_pos').double_value
        closed_pos = P('gripper.closed_pos').double_value
        joint_names = [s for s in P('gripper.joint_names').string_array_value]

        r = np.clip(width_m / max_w, 0.0, 1.0)  # 0 closed .. 1 open (by width)
        pos = closed_pos + r * (open_pos - closed_pos)

        if len(joint_names) == 1:
            return joint_names, [float(pos)]
        elif len(joint_names) == 2:
            # mirror same pos to both joints (common for parallel grippers)
            return joint_names, [float(pos), float(pos)]
        else:
            # fallback: broadcast to all provided joints
            return joint_names, [float(pos)] * len(joint_names)

    # ----------------- Grasp -> MoveIt2 conversion -----------------
    def make_moveit_grasp(self, T_4x4: np.ndarray, width_m: float, score: float, idx: int) -> Grasp:
        """
        Convert one SE(3) grasp + width into moveit_msgs/msg/Grasp.
        - Pose is in the configured frame_id (assumed camera frame).
        - Pre-grasp posture uses 'open'; grasp posture uses 'closed'.
        - Approach vector is along +Z of the grasp (approach direction).
        """
        P = lambda n: self.get_parameter(n).get_parameter_value()
        frame_id = P('frame_id').string_value

        # Build PoseStamped from T
        gpose = PoseStamped()
        gpose.header.frame_id = frame_id
        gpose.header.stamp = self.get_clock().now().to_msg()

        R = T_4x4[:3, :3]
        t = T_4x4[:3, 3]
        qx, qy, qz, qw = rotmat_to_quat_xyzw(R)

        gpose.pose.position.x = float(t[0])
        gpose.pose.position.y = float(t[1])
        gpose.pose.position.z = float(t[2])
        gpose.pose.orientation.x = float(qx)
        gpose.pose.orientation.y = float(qy)
        gpose.pose.orientation.z = float(qz)
        gpose.pose.orientation.w = float(qw)

        # Pre-grasp (open) & grasp (close) postures
        joint_names, open_positions = self.width_to_joint_positions(P('gripper.max_width_m').double_value)
        _, close_positions = self.width_to_joint_positions(0.0)

        pre = JointTrajectory()
        pre.joint_names = joint_names
        pt_open = JointTrajectoryPoint()
        pt_open.positions = open_positions
        pt_open.time_from_start = Duration(sec=0, nanosec=500_000_000)  # 0.5s
        pre.points = [pt_open]

        grasp_post = JointTrajectory()
        grasp_post.joint_names = joint_names
        pt_close = JointTrajectoryPoint()
        pt_close.positions = close_positions
        pt_close.time_from_start = Duration(sec=0, nanosec=700_000_000)  # 0.7s
        grasp_post.points = [pt_close]

        # Approach / Retreat along grasp Z-axis
        z_axis = R[:, 2]  # approach direction per our convention
        app = GripperTranslation()
        app.direction.header.frame_id = frame_id
        app.direction.vector.x = float(z_axis[0])
        app.direction.vector.y = float(z_axis[1])
        app.direction.vector.z = float(z_axis[2])
        app.min_distance = P('approach_min_dist').double_value
        app.desired_distance = P('approach_desired').double_value

        ret = GripperTranslation()
        ret.direction.header.frame_id = frame_id
        # retreat opposite to approach
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

    # ----------------- Service handler -----------------
    def handle_get_top_grasps(self, request, response):
        P = lambda n: self.get_parameter(n).get_parameter_value()
        rgb_path = P('rgb_path').string_value
        depth_path = P('depth_path').string_value

        out = self.run_inference_once(rgb_path, depth_path)
        if out is None:
            response.success = False
            response.message = 'No grasps found.'
            return response

        T_mats = out['T_mats']
        widths = out['widths']
        scores = out['scores']

        # Convert and publish each Grasp
        for i in range(len(scores)):
            grasp_msg = self.make_moveit_grasp(T_mats[i], float(widths[i]), float(scores[i]), i)
            self.grasp_pub.publish(grasp_msg)

        response.success = True
        response.message = f'Published {len(scores)} grasps to /top_grasps'
        return response


# --------------- main ---------------
def main():
    rclpy.init()
    node = GraspPoseService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
