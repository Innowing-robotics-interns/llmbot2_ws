import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/fyp/llmbot2_ws/src/sem_map/install/sem_map'
