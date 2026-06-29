#!/usr/bin/env python3
"""
gui_bringup.launch.py  —  Anveshan GCS, the GUI half, in one command.
====================================================================
Brings up EVERYTHING the GUI team owns:

    semantic_indexer  +  semantic_retriever  +  hd_frame_server      (ROS 2 nodes)
    FastAPI backend (port 5000)  +  static frontend (port 8080)      (web stack)

This replaces the 4–5 manual terminals used during bring-up. It is intentionally
NOT a ros2 package launch file — iroc_gui is plain backend/frontend — so run it
BY PATH:

    # 1) activate the venv that has torch + transformers, then source the ws
    source ~/seed_searcher_naive/img_p_new/bin/activate
    cd ~/anveshan_ws && source install/setup.bash

    # 2) launch (example: replay the existing test_bag31 DB through the GUI)
    ros2 launch /path/to/iroc_gui/launch/gui_bringup.launch.py \
        rosbag_path:=/home/nidar/Desktop/test_bag31 \
        camera_topic:=/image_raw \
        db_path:=/home/nidar/semantic_db/test_bag31.pt \
        seeds_dir:=/home/nidar/anveshan_seeds \
        with_indexer:=false            # DB already built → skip the indexer

Why ExecuteProcess + `python3 -m`, not Node():
  The console-script wrappers get a system-python shebang at build time, and
  system python has no `transformers`. Spawning `python3 -m semantic_retrieval.X`
  uses whatever `python_exe` resolves to — i.e. the ACTIVE VENV python when you
  launch from an activated venv. No rebuild, no shebang surgery.

Startup ordering is not critical by design:
  - retriever polls for the DB file and load-on-appear;
  - hd_frame_server builds its frame index lazily on first request;
  - the SemanticMatch results topic is LATCHED, so the backend gets the last
    result the instant it connects, regardless of who started first.

Distributed deployment (Jetson nodes + laptop GUI on the same ROS_DOMAIN_ID):
  - On the Jetson:  with_backend:=false with_frontend:=false
  - On the laptop:  with_indexer:=false with_retriever:=false with_hd_server:=false
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # iroc_gui/ is the parent of this launch/ folder (file is run by path).
    iroc_gui_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # config.yaml ships in the installed semantic_retrieval package share.
    sr_share = get_package_share_directory('semantic_retrieval')
    config = os.path.join(sr_share, 'config', 'config.yaml')

    # ---- Launch arguments -------------------------------------------------
    args = [
        DeclareLaunchArgument(
            'python_exe', default_value='python3',
            description='Interpreter for the ML nodes + web server. Defaults to '
                        'python3 (= the venv python when launched from an active '
                        'venv). Or pass an absolute venv path.'),
        DeclareLaunchArgument(
            'camera_topic', default_value='/camera/camera/color/image_raw',
            description='Camera image topic in the rosbag. For test_bag31 use /image_raw.'),
        DeclareLaunchArgument(
            'rosbag_path', default_value='',
            description='Mission rosbag DIRECTORY (the one with metadata.yaml). '
                        'Required by hd_frame_server and by indexing.'),
        DeclareLaunchArgument(
            'db_path', default_value=os.path.expanduser('~/semantic_db/mission.pt'),
            description='Embedding DB path: indexer writes it, retriever reads it.'),
        DeclareLaunchArgument(
            'seeds_dir', default_value=os.path.expanduser('~/anveshan_seeds'),
            description='Folder of seed images loaded at startup (the GUI can also '
                        'add seeds live over /semantic_retrieval/add_seed).'),
        DeclareLaunchArgument(
            'threshold', default_value='0.0',
            description='Retrieval FLOOR — the lower bound of the candidate pool the '
                        'retriever computes. Default 0.0 = run the whole search; the '
                        'GUI then thresholds live within the pool. Raise it only to '
                        'cap how much is published.'),
        DeclareLaunchArgument(
            'display_threshold', default_value='0.57',
            description='The DEFAULT threshold the GUI shows on load (the numbox '
                        'value). Set this from the terminal to fix the operating '
                        'point for an autonomous run; the operator can still change '
                        'it live in the GUI for a manual demo.'),
        DeclareLaunchArgument(
            'top_k', default_value='50',
            description='Max candidate frames per seed in the pool (paired with the '
                        '0.0 floor for live GUI thresholding).'),
        DeclareLaunchArgument(
            'downsample_seed', default_value='false',
            description='If true, pyramid-downsample each seed to 128x128 before '
                        'embedding (for full-res seed photos). false = seed is '
                        'assumed already a 128x128 downsampled crop.'),
        DeclareLaunchArgument(
            'sample_rate', default_value='1',
            description='Indexer frame sampling (1 = embed every frame).'),
        DeclareLaunchArgument(
            'frontend_port', default_value='8080',
            description='Port for the static frontend server.'),
        DeclareLaunchArgument(
            'gcs_token', default_value='',
            description='If non-empty, exported as GCS_COMMAND_TOKEN so the backend '
                        'enforces X-GCS-Token auth on the override/seed endpoints.'),
        # Component toggles — flip to split Jetson (nodes) vs laptop (GUI).
        DeclareLaunchArgument('with_indexer',   default_value='true'),
        DeclareLaunchArgument('with_retriever', default_value='true'),
        DeclareLaunchArgument('with_hd_server', default_value='true'),
        DeclareLaunchArgument('with_backend',   default_value='true'),
        DeclareLaunchArgument('with_frontend',  default_value='true'),
    ]

    py = LaunchConfiguration('python_exe')

    # DINOv2 weights are cached on-device → force transformers offline so a
    # node never blocks on a network reach for the model.
    ml_env = {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}

    # ---- Semantic nodes (venv python via `-m`) ----------------------------
    indexer = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('with_indexer')),
        cmd=[py, '-m', 'semantic_retrieval.indexer_node', '--ros-args',
             '--params-file', config,
             '-p', ['camera_topic:=', LaunchConfiguration('camera_topic')],
             '-p', ['sample_rate:=', LaunchConfiguration('sample_rate')],
             '-p', ['embedding_db_path:=', LaunchConfiguration('db_path')]],
        cwd=iroc_gui_dir, additional_env=ml_env, output='screen',
        name='semantic_indexer')

    retriever = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('with_retriever')),
        cmd=[py, '-m', 'semantic_retrieval.retriever_node', '--ros-args',
             '--params-file', config,
             '-p', ['embedding_db_path:=', LaunchConfiguration('db_path')],
             '-p', ['seeds_dir:=', LaunchConfiguration('seeds_dir')],
             '-p', ['similarity_threshold:=', LaunchConfiguration('threshold')],
             '-p', ['top_k:=', LaunchConfiguration('top_k')],
             '-p', ['downsample_seed:=', LaunchConfiguration('downsample_seed')]],
        cwd=iroc_gui_dir, additional_env=ml_env, output='screen',
        name='semantic_retriever')

    hd_server = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('with_hd_server')),
        cmd=[py, '-m', 'semantic_retrieval.hd_frame_server', '--ros-args',
             '--params-file', config,
             '-p', ['camera_topic:=', LaunchConfiguration('camera_topic')],
             '-p', ['rosbag_path:=', LaunchConfiguration('rosbag_path')]],
        cwd=iroc_gui_dir, additional_env=ml_env, output='screen',
        name='hd_frame_server')

    # ---- Web stack --------------------------------------------------------
    backend = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('with_backend')),
        cmd=[py, 'backend/backend.py'],
        cwd=iroc_gui_dir,
        additional_env={
            'GCS_COMMAND_TOKEN': LaunchConfiguration('gcs_token'),
            # Lets the backend serve seed thumbnails straight from the seeds_dir
            # when it shares the filesystem (all-on-one-box). Harmless if the path
            # doesn't exist (e.g. backend on a separate laptop).
            'GCS_SEEDS_DIR': LaunchConfiguration('seeds_dir'),
            # The GUI's default threshold box value (served via /api/config).
            'GCS_DISPLAY_THRESHOLD': LaunchConfiguration('display_threshold'),
        },
        output='screen', name='gcs_backend')

    frontend = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('with_frontend')),
        cmd=[py, '-m', 'http.server', LaunchConfiguration('frontend_port'),
             '-d', 'frontend'],
        cwd=iroc_gui_dir,
        output='screen', name='gcs_frontend')

    banner = LogInfo(msg=[
        '\n┌─ Anveshan GCS bring-up ───────────────────────────────────\n',
        '│ backend  : http://0.0.0.0:5000  (docs /docs)\n',
        '│ frontend : http://0.0.0.0:', LaunchConfiguration('frontend_port'), '\n',
        '│ rosbag   : ', LaunchConfiguration('rosbag_path'), '\n',
        '│ db       : ', LaunchConfiguration('db_path'), '\n',
        '│ camera   : ', LaunchConfiguration('camera_topic'), '\n',
        '│ retrieval: floor ', LaunchConfiguration('threshold'),
        ' | top_k ', LaunchConfiguration('top_k'),
        ' | GUI default thr ', LaunchConfiguration('display_threshold'), '\n',
        '└───────────────────────────────────────────────────────────',
    ])

    return LaunchDescription(
        args + [banner, indexer, retriever, hd_server, backend, frontend])
