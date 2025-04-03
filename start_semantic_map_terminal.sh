#!/bin/bash

gnome-terminal --tab --title="sem_map_lseg_feat" -- bash -c "ros2 run sem_map sem_map_service_lseg_feat"
gnome-terminal --tab --title="sem_map_yolo_lseg" -- bash -c "ros2 run sem_map sem_map_service_yolo_lseg"

sleep 10

gnome-terminal --tab --title="image_socket_recv" -- bash -c "ros2 run sem_map image_socket_recv"
gnome-terminal --tab --title="query_socket_handler" -- bash -c "ros2 run sem_map query_socket_handler"

sleep 10

echo "all sessions started, some may fail, check the successful launches"

# tmux list-sessions