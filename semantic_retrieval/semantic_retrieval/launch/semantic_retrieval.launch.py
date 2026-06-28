import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('semantic_retrieval')
    config = os.path.join(pkg_share, 'config', 'config.yaml')

    common = dict(package='semantic_retrieval', output='screen', parameters=[config])

    return LaunchDescription([
        # Phase 1 — builds the embedding DB from the rosbag (triggered by service)
        Node(executable='indexer_node',   name='semantic_indexer',  **common),
        # Phase 2 — loads DB, matches seeds, publishes SemanticMatch metadata
        Node(executable='retriever_node', name='semantic_retriever', **common),
        # Phase 3 — serves HD frames from the rosbag on demand (GetHdFrame srv)
        Node(executable='hd_frame_server', name='hd_frame_server',   **common),
    ])
