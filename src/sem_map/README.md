Semantic Map is written as a ROS service, clients can query the semantic map by sending query in format of: llmbot2_ws/src/interfaces/srv/SemanticQuery.srv
To launch teh semantic map service
```bash
ros2 run sem_map_service_***
```
where sem_map_service_*** is could be one of the following:
```bash
sem_map_service_cyw1_lseg
sem_map_service_gd_lseg
sem_map_service_lseg_feat
sem_map_service_yolo_lseg
sem_map_service_yw_lseg
```
(I tried writing it in one launch file and set the model as parameter, but the guide in ROS official website didn't work, if you are later interns you could try fixing it)

If you are using https://github.com/BillieGuo/HKU-2425FYP-RobotHelper
First run Semantic Map socket on robot
   ```bash
    ros2 run semantic_map image_transform_listener
    ```
Then in llmbot2_ws run Semantic Map Server
    ```bash
    cd llmbot2_ws
    ./start_semantic_map.sh
    # Wait until the tmux sessions are listed
    # in another terminal, to enable gdino explore
    ros2 run sem_map gdino_query_client
    ```
Then go back to the robot's on board computer Semantic Map Query
    ```bash
    ros2 run semantic_map query_semantic_map
    ```
