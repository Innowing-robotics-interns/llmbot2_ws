#!/bin/bash

echo "Starting tmux sessions"

tmux new-session -d -s sem_map_lseg_feat "ros2 run sem_map sem_map_service_lseg_feat" 
tmux new-session -d -s sem_map_yolo_lseg "ros2 run sem_map sem_map_service_yolo_lseg" 

sleep 10

tmux new-session -d -s image_transform_client.py "ros2 run sem_map image_transform_client.py" 
tmux new-session -d -s query_socket_handler "ros2 run sem_map query_socket_handler" 

echo "Sessions Started:"
tmux list-sessions
