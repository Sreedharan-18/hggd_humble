#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Defaults; override on CLI only if needed
    hggd_root         = DeclareLaunchArgument('hggd_root',            default_value='/ros2_ws/src/hggd_ros2/HGGD')
    checkpoint_path   = DeclareLaunchArgument('checkpoint_path',      default_value='/ros2_ws/src/hggd_ros2/HGGD/realsense_checkpoint')
    rgb_topic         = DeclareLaunchArgument('rgb_topic',            default_value='/wrist_mounted_camera/image')
    depth_topic       = DeclareLaunchArgument('depth_topic',          default_value='/wrist_mounted_camera/depth_image')
    camera_info_topic = DeclareLaunchArgument('camera_info_topic',    default_value='/wrist_mounted_camera/camera_info')
    frame_id          = DeclareLaunchArgument('frame_id',             default_value='color_optical_frame')

    input_h           = DeclareLaunchArgument('input_h',              default_value='360')
    input_w           = DeclareLaunchArgument('input_w',              default_value='640')
    anchor_num        = DeclareLaunchArgument('anchor_num',           default_value='7')
    all_points_num    = DeclareLaunchArgument('all_points_num',       default_value='25600')
    center_num        = DeclareLaunchArgument('center_num',           default_value='48')
    group_num         = DeclareLaunchArgument('group_num',            default_value='512')
    local_k           = DeclareLaunchArgument('local_k',              default_value='10')
    ratio             = DeclareLaunchArgument('ratio',                default_value='8')
    local_thres       = DeclareLaunchArgument('local_thres',          default_value='0.01')
    heatmap_thres     = DeclareLaunchArgument('heatmap_thres',        default_value='0.01')
    topk_per_object   = DeclareLaunchArgument('topk_per_object',      default_value='5')

    latched           = DeclareLaunchArgument('latched',              default_value='true')
    repeat_hz         = DeclareLaunchArgument('repeat_hz',            default_value='0.0')

    viz_enable        = DeclareLaunchArgument('viz_enable',           default_value='true')
    viz_axis_len      = DeclareLaunchArgument('viz_axis_len',         default_value='0.08')
    viz_axis_diam     = DeclareLaunchArgument('viz_axis_diam',        default_value='0.01')

    gripper_joint_names = DeclareLaunchArgument('gripper.joint_names', default_value='[gripper_left_joint,gripper_right_joint]')
    gripper_max_width   = DeclareLaunchArgument('gripper.max_width_m', default_value='0.10')
    gripper_open_pos    = DeclareLaunchArgument('gripper.open_pos',    default_value='0.0')
    gripper_closed_pos  = DeclareLaunchArgument('gripper.closed_pos',  default_value='0.8')

    approach_min_dist = DeclareLaunchArgument('approach_min_dist', default_value='0.05')
    approach_desired  = DeclareLaunchArgument('approach_desired',  default_value='0.10')
    retreat_min_dist  = DeclareLaunchArgument('retreat_min_dist',  default_value='0.05')
    retreat_desired   = DeclareLaunchArgument('retreat_desired',   default_value='0.10')

    node = Node(
        package='grasp_pose_detection',
        executable='hggd_grasp_service',   # console_scripts entry points to grasp_pose_service:main
        name='grasp_pose_service_topics_per_object',
        output='screen',
        parameters=[{
            'hggd_root':             LaunchConfiguration('hggd_root'),
            'checkpoint_path':       LaunchConfiguration('checkpoint_path'),
            'rgb_topic':             LaunchConfiguration('rgb_topic'),
            'depth_topic':           LaunchConfiguration('depth_topic'),
            'camera_info_topic':     LaunchConfiguration('camera_info_topic'),
            'frame_id':              LaunchConfiguration('frame_id'),
            'input_h':               LaunchConfiguration('input_h'),
            'input_w':               LaunchConfiguration('input_w'),
            'anchor_num':            LaunchConfiguration('anchor_num'),
            'all_points_num':        LaunchConfiguration('all_points_num'),
            'center_num':            LaunchConfiguration('center_num'),
            'group_num':             LaunchConfiguration('group_num'),
            'local_k':               LaunchConfiguration('local_k'),
            'ratio':                 LaunchConfiguration('ratio'),
            'local_thres':           LaunchConfiguration('local_thres'),
            'heatmap_thres':         LaunchConfiguration('heatmap_thres'),
            'topk_per_object':       LaunchConfiguration('topk_per_object'),
            'latched':               LaunchConfiguration('latched'),
            'repeat_hz':             LaunchConfiguration('repeat_hz'),
            'viz_enable':            LaunchConfiguration('viz_enable'),
            'viz_axis_len':          LaunchConfiguration('viz_axis_len'),
            'viz_axis_diam':         LaunchConfiguration('viz_axis_diam'),
            'gripper.joint_names':   LaunchConfiguration('gripper.joint_names'),
            'gripper.max_width_m':   LaunchConfiguration('gripper.max_width_m'),
            'gripper.open_pos':      LaunchConfiguration('gripper.open_pos'),
            'gripper.closed_pos':    LaunchConfiguration('gripper.closed_pos'),
            'approach_min_dist':     LaunchConfiguration('approach_min_dist'),
            'approach_desired':      LaunchConfiguration('approach_desired'),
            'retreat_min_dist':      LaunchConfiguration('retreat_min_dist'),
            'retreat_desired':       LaunchConfiguration('retreat_desired'),
        }],
    )

    return LaunchDescription([
        hggd_root, checkpoint_path, rgb_topic, depth_topic, camera_info_topic, frame_id,
        input_h, input_w, anchor_num, all_points_num, center_num, group_num,
        local_k, ratio, local_thres, heatmap_thres, topk_per_object,
        latched, repeat_hz,
        viz_enable, viz_axis_len, viz_axis_diam,
        gripper_joint_names, gripper_max_width, gripper_open_pos, gripper_closed_pos,
        approach_min_dist, approach_desired, retreat_min_dist, retreat_desired,
        node
    ])
