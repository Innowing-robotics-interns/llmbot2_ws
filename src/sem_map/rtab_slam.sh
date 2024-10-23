#!/bin/bash
ros2 launch rtabmap_launch rtabmap.launch.py \
    rtabmap_args:="--delete_db_on_start --Optimizer/GravitySigma 0.3" \
    depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
    rgb_topic:=/camera/camera/color/image_raw \
    camera_info_topic:=/camera/camera/color/camera_info \
    approx_sync:=false \
    wait_imu_to_init:=true \
    imu_topic:=/rtabmap/imu
