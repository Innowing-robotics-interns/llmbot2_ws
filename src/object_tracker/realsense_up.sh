#!/bin/bash
# ros2 launch realsense2_camera rs_launch.py depth_module.depth_profile:=1280x720x30 pointcloud.enable:=true enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true

