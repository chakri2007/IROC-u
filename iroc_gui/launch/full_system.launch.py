#!/usr/bin/env python3
"""
full_system.launch.py  —  FINAL-TEST skeleton (the "universal" launcher).
=========================================================================
Brings up the GUI half (via gui_bringup.launch.py) and leaves clearly-marked
slots for the FLIGHT half. The flight half lives in iroc_ros_ws / precision_landing
/ vio — it is owned by the flight-code team and is READ-ONLY to the GUI team, so
this file keeps it as commented placeholders rather than hard wiring.

INTENT: this file will be edited (a lot) by the flight team as their packages
firm up. Treat it as a starting template, not a finished artifact. The piece that
is real and maintained is gui_bringup.launch.py; everything below the GUI include
is a stub for the team to fill in.

Run by path (after sourcing ROS + the workspace + venv):
    ros2 launch /path/to/iroc_gui/launch/full_system.launch.py \
        rosbag_path:=/path/to/mission_bag

Make sure every machine in the test shares one ROS_DOMAIN_ID.
"""

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch_ros.actions import Node  # uncomment when wiring flight nodes


def generate_launch_description():
    here = os.path.dirname(os.path.abspath(__file__))

    # --- GUI half (real, maintained) --------------------------------------
    # Pass-through: any arg you give full_system is forwarded to gui_bringup
    # by re-declaring it here in launch_arguments, e.g.:
    #   launch_arguments={'rosbag_path': LaunchConfiguration('rosbag_path')}.items()
    gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(here, 'gui_bringup.launch.py')),
    )

    return LaunchDescription([
        LogInfo(msg='[full_system] GUI half up. Flight half is team-owned — see stubs below.'),
        gui,

        # ══════════════════════════════════════════════════════════════════
        # FLIGHT HALF — owned by the flight-code team. Wire these in here.
        # (Do NOT edit the packages themselves from the GUI side; just include
        #  their launch files / nodes below.)
        #
        #  1) MAVROS  — FCU link, /mavros/* telemetry the GUI Controller reads:
        #  IncludeLaunchDescription(PythonLaunchDescriptionSource(
        #      os.path.join(get_package_share_directory('mavros'), 'launch', 'px4.launch'))),
        #
        #  2) VIO / VSLAM — publishes /visual_slam/tracking/odometry.
        #     REQUIRED for real XYZ on the Analysis cards (bench bags without it
        #     show pose "unavailable"):
        #  IncludeLaunchDescription(PythonLaunchDescriptionSource('.../vslam.launch.py')),
        #
        #  3) Precision landing / mission state machine (undock→survey→land→dock).
        #     This is also what TRIGGERS indexing after docking, via:
        #       /semantic_retrieval/trigger_indexing  (TriggerIndexing.srv)
        #  IncludeLaunchDescription(PythonLaunchDescriptionSource('.../mission.launch.py')),
        #
        #  4) GCS command bridge — subscribes /gcs/command (START/ABORT/HOLD/...)
        #     and publishes /gcs/command_ack. Full contract + ready node:
        #     iroc_gui/HANDOFF.md. QoS MUST be RELIABLE / VOLATILE / depth 10.
        #  Node(package='<their_pkg>', executable='gcs_command_bridge', output='screen'),
        #
        #  5) Docking-station / TCP comms bring-up, if it runs on this machine.
        # ══════════════════════════════════════════════════════════════════
    ])
