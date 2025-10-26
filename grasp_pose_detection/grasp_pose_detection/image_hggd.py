#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge


def _default_outdir() -> Path:
    """Prefer $HGGD_ROOT/images if available, else ./images."""
    hggd = os.environ.get("HGGD_ROOT")
    if hggd:
        return (Path(hggd) / "images").resolve()
    return (Path.cwd() / "images").resolve()


def _stamp_to_sec(stamp) -> float:
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


class RGBDSavePNG(Node):
    """
    Subscribe to RGB + Depth, sync by timestamp, save two PNGs:
      - images/ros2_rgb.png    (uint8, 3-channel, BGR on disk via OpenCV)
      - images/ros2_depth.png  (uint16, single-channel, millimeters)
    Optionally resize both to a target WxH (default: 1280x720).
    """

    def __init__(self) -> None:
        super().__init__("rgbd_save_png")
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter("rgb_topic", "/wrist_mounted_camera/image")
        self.declare_parameter("depth_topic", "/wrist_mounted_camera/depth_image")
        self.declare_parameter("output_dir", str(_default_outdir()))
        self.declare_parameter("stamp_tolerance_ms", 80.0)
        self.declare_parameter("timeout_sec", 30.0)
        self.declare_parameter("depth_clip_max_m", 10.0)

        # Resize control (set to your demo’s expected size 1280x720)
        self.declare_parameter("force_resize", True)
        self.declare_parameter("target_width", 1280)
        self.declare_parameter("target_height", 720)

        p = lambda n: self.get_parameter(n).get_parameter_value()
        self.rgb_topic = p("rgb_topic").string_value
        self.depth_topic = p("depth_topic").string_value
        self.out_dir = Path(p("output_dir").string_value).expanduser()
        self.tol = float(p("stamp_tolerance_ms").double_value) / 1000.0
        self.timeout_sec = float(p("timeout_sec").double_value)
        self.depth_clip_max_m = float(p("depth_clip_max_m").double_value)
        self.force_resize = bool(p("force_resize").bool_value)
        self.tgt_w = int(p("target_width").integer_value)
        self.tgt_h = int(p("target_height").integer_value)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_rgb = self.out_dir / "ros2_rgb.png"
        self.out_depth = self.out_dir / "ros2_depth.png"

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(ROSImage, self.rgb_topic, self._rgb_cb, qos)
        self.create_subscription(ROSImage, self.depth_topic, self._depth_cb, qos)

        self.last_rgb: Optional[Tuple[ROSImage, np.ndarray]] = None  # (msg, HxWx3 uint8 RGB)
        self.last_depth: Optional[Tuple[ROSImage, np.ndarray]] = None  # (msg, HxW float meters)

        self._t0 = time.time()
        self.create_timer(0.5, self._watchdog)

        self.get_logger().info(f"Waiting for synchronized frames within {self.tol*1000:.0f} ms…")
        self.get_logger().info(f"RGB  topic : {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Output dir : {self.out_dir}")
        if self.force_resize:
            self.get_logger().info(f"Will resize to {self.tgt_w}x{self.tgt_h}")

    # Callbacks
    def _rgb_cb(self, msg: ROSImage):
        try:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")  # HxWx3 uint8
            self.last_rgb = (msg, rgb)
            self._maybe_save()
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")

    def _depth_cb(self, msg: ROSImage):
        try:
            d = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            arr = np.asarray(d)

            # Normalize to float meters
            if arr.dtype == np.uint16:
                d_m = arr.astype(np.float32) / 1000.0
            elif arr.dtype in (np.float32, np.float64):
                d_m = arr.astype(np.float32)
            else:
                self.get_logger().warn(f"Unexpected depth dtype {arr.dtype}; assuming millimeters")
                d_m = arr.astype(np.float32) / 1000.0

            d_m = np.nan_to_num(d_m, nan=0.0, posinf=0.0, neginf=0.0)
            if self.depth_clip_max_m > 0:
                d_m = np.clip(d_m, 0.0, self.depth_clip_max_m)

            self.last_depth = (msg, d_m)
            self._maybe_save()
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    # Save when both frames are present and synced
    def _maybe_save(self):
        if self.last_rgb is None or self.last_depth is None:
            return

        m_rgb, rgb = self.last_rgb
        m_d, d_m = self.last_depth
        t_rgb = _stamp_to_sec(m_rgb.header.stamp)
        t_d = _stamp_to_sec(m_d.header.stamp)

        if abs(t_rgb - t_d) > self.tol:
            # keep the newer frame; drop the older so we converge quickly
            if t_rgb > t_d:
                self.last_depth = None
            else:
                self.last_rgb = None
            return

        # Resize if requested (RGB: linear; Depth: nearest to preserve mm)
        if self.force_resize:
            rgb = cv2.resize(rgb, (self.tgt_w, self.tgt_h), interpolation=cv2.INTER_LINEAR)
            d_m = cv2.resize(d_m, (self.tgt_w, self.tgt_h), interpolation=cv2.INTER_NEAREST)

        # Save RGB (convert to BGR for OpenCV)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok_rgb = cv2.imwrite(str(self.out_rgb), rgb_bgr)

        # Save Depth as 16-bit millimeters
        d_mm = np.clip(d_m * 1000.0, 0, 65535).astype(np.uint16)
        ok_depth = cv2.imwrite(str(self.out_depth), d_mm)

        if ok_rgb and ok_depth:
            self.get_logger().info(f"Saved:\n  {self.out_rgb}\n  {self.out_depth}")
            self.get_logger().info(
                f"Depth stats (mm): min={int(d_mm.min())} max={int(d_mm.max())} "
                f"shape={d_mm.shape} dtype={d_mm.dtype}"
            )
        else:
            if not ok_rgb:
                self.get_logger().error(f"Failed to write {self.out_rgb}")
            if not ok_depth:
                self.get_logger().error(f"Failed to write {self.out_depth}")

        rclpy.shutdown()  # one-shot

    def _watchdog(self):
        if (time.time() - self._t0) > self.timeout_sec:
            self.get_logger().error("Timed out waiting for synchronized frames.")
            rclpy.shutdown()


def main():
    rclpy.init()
    node = RGBDSavePNG()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if rclpy.ok():
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
