import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='sem_map',
            executable='update_map',
            name='update_map'),
  ])