from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    image_processor_arg = DeclareLaunchArgument(
        'image_processor',  # 使用哪个图片处理器
        default_value='lseg_feat',
        description='The image processor to use, options: lseg_feat, yolo_lseg,'  # 可以选择什么
    )
    node = Node(
        package='sem_map',
        # namespace='sem_map_srv',
        executable='sem_map_service',
        name='sem_map_service',
        parameters=[
            {'image_processor': LaunchConfiguration('image_processor')}
        ]
    )
    return LaunchDescription([
        image_processor_arg,
        node
    ])