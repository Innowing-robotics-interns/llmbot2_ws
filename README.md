# llmbot2_ws
ros2 workspace for a llm-driven robot (object tracekr, semantic map ..
with additional hardware including jetson orin, a 4 wheel rover, mid 360 lidar
and additional packages to run localization, navigation, and LLM command control: https://github.com/Innowing-robotics-interns/Rover_Official
## object_tracker
- locate object's mask in the first frame and track the object's center at each frame
- owl_vit -> segment anything -> XMEM -> depth deptroject to 3D
- hugging face owl vit: https://huggingface.co/docs/transformers/model_doc/owlvit
- segment anything: https://github.com/facebookresearch/segment-anything
- XMEM: https://github.com/hkchengrex/XMem
#### **video demo**(Click to view)
[![Watch the video](https://img.youtube.com/vi/PtvrjFVf8sE/0.jpg)](https://www.youtube.com/watch?v=PtvrjFVf8sE)
## sem_map
- a replication of VLMaps(https://github.com/vlmaps/vlmaps) with some custom modifications
- building 3D semantic map, input the object name and find its' 3D location
- LSeg encoding -> cluster to few points -> deproject to 3D
#### **video demo**(Click to view)

[![Watch the video](https://img.youtube.com/vi/3x6oQzM47_Q/0.jpg)](https://www.youtube.com/watch?v=3x6oQzM47_Q)

#### **long video demo, includes building map**(Click to view)

[![Watch the video](https://img.youtube.com/vi/KzZGFgizNwY/0.jpg)](https://www.youtube.com/watch?v=KzZGFgizNwY)

