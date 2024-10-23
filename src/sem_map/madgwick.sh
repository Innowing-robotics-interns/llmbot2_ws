#!/bin/bash
# ros2 launch imu_filter_madgwick imu_filter.launch.py -- _use_mag:=false _publish_tf:=false _world_frame:="enu" /imu/data_raw:=/camera/camera/imu /imu/data:=/rtabmap/imu
ros2 launch imu_filter_madgwick imu_filter.launch.py
