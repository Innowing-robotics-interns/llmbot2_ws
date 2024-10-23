#!/bin/bash
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true enable_gyro:=true enable_accel:=true unite_imu_method:=2
