#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import time
import math
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import TwistStamped, PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from apriltag_msgs.msg import AprilTagDetectionArray
from std_msgs.msg import Int32

import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class AutonomousPrecisionLanding(Node):

    def __init__(self):
        super().__init__('autonomous_precision_landing')

        # ================== QOS & STATE ==================
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )

        self.state = State()
        self.local_pos = PoseStamped()
        self.initial_alt = 0.0

        # Tracking State
        self.tag_frame = ""
        self.err_x = 0.0
        self.err_y = 0.0
        self.err_z = 0.0

        # PID memory (original simple PD)
        self.prev_err_x = 0.0
        self.prev_err_y = 0.0
        self.prev_time = time.time()

        # ================== PARAMETERS ==================
        self.declare_parameter("takeoff_alt", 2.0)
        self.declare_parameter("kp", 0.5)
        self.declare_parameter("kd", 0.0)
        self.declare_parameter("max_vel_xy", 0.3)
        self.declare_parameter("land_alt_threshold", 0.2)

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.camera_frame = "camera_color_optical_frame"

        # ================== COMMS ==================
        self.create_subscription(State, '/mavros/state', self.state_cb, qos)
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.pos_cb, qos)
        self.create_subscription(AprilTagDetectionArray, '/detections', self.aruco_cb, qos_profile_sensor_data)

        # FIXED MESSAGE TYPE
        self.create_subscription(Int32, '/mission_initiate', self.mission_initiate_cb, 10)

        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)

        # Services
        self.arm_srv = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.mode_srv = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_srv = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # Mission State
        self.mission_initiated = False

        # ================== AUTO TUNING INTEGRATION (EXACT LOGIC FROM DroneTunerNode) ==================
        # Only X/Y PID is applied (Z velocity remains 0 during tuning and uses original descent logic after).
        # All core tuning flags, gain adjustment rules, delta-time logic, buffers, and PID formula are 100% unchanged.
        # Error input to PID comes from AprilTag relative pose (single fixed target = tag center).
        # After tuning completes (flag_Kp_start == 3 + 3s stable), descent begins with tuned gains.
        self.tuner_err_x = [0.0] * 54
        self.tuner_err_y = [0.0] * 54
        self.tuner_err_z = [0.0] * 54
        self.err_x_previous = [0.0] * 18
        self.err_y_previous = [0.0] * 18
        self.err_z_previous = [0.0] * 18
        self.err_x_auto = 0.0
        self.err_y_auto = 0.0
        self.err_z_auto = 0.0
        self.err_prior_x_auto = 0.0
        self.err_prior_y_auto = 0.0
        self.err_prior_z_auto = 0.0
        self.err = [0.0] * 18
        self.err_prior = [0.0] * 18
        self.out = [0.0, 0.0, 0.0]
        self.now_time = 0.0
        self.prev_time = time.time()
        self.prev_whycon = time.time()
        self.now_whycon = 0.0
        self.prev_ki = time.time()
        self.now_ki = 0.0
        self.start_time_tuning = 0.0
        self.flag_initialize = 0
        self.a = 0
        self.w = 0
        self.r = 0
        self.flag_Kp_start = 0
        self.count = 0
        self.j = 0
        self.i = 0
        self.timeout1 = 0.0
        self.timeout = 0.0
        self.stable_start_time = None
        self.sumerr = [0.0] * 18
        self.sumerr1 = [0.0] * 18
        self.max_vel = 120
        self.min_vel = -120
        self.max_vel_z = 250
        self.min_vel_z = -250
        self.Kp = [0.0, 0.0, 0.0]
        self.Ki = [0.0, 0.0, 0.0]
        self.Kd = [0.0, 0.0, 0.0]
        self.Kp_max = [0.0, 0.0, 0.0]
        self.Kp_min = [0.0, 0.0, 0.0]
        self.Kd_max = [0.0, 0.0, 0.0]
        self.Kd_min = [0.0, 0.0, 0.0]
        self.Ki_max = [0.0, 0.0, 0.0]
        self.Ki_min = [0.0, 0.0, 0.0]
        self.kp_auto_x = 0.0
        self.kp_auto_y = 0.0
        self.kp_auto_z = 0.0
        self.kd_auto_x = 0.0
        self.kd_auto_y = 0.0
        self.kd_auto_z = 0.0
        self.ki_auto_x = 0.0
        self.ki_auto_y = 0.0
        self.ki_auto_z = 0.0
        # dummy list_co (kept for exact compatibility with original motion/PID structure)
        self.list_co = [[0, 0, 0]]

        # Tuning state flags (exact flow preserved)
        self.tuning_started = False
        self.tuning_complete = False

        self.get_logger().info('AutonomousPrecisionLanding node initialized with exact auto-tuning logic merged.')

    # -------------------- Callbacks (UNTOUCHED) --------------------
    def mission_initiate_cb(self, msg):
        if msg.data == 1 and not self.mission_initiated:
            self.get_logger().info("Mission Initiation Received!")
            self.mission_initiated = True

    def state_cb(self, msg):
        self.state = msg

    def pos_cb(self, msg):
        self.local_pos = msg

    def aruco_cb(self, msg):
        if len(msg.detections) > 0:
            det = msg.detections[0]
            tag_id = det.id[0] if hasattr(det.id, '__iter__') else det.id
            self.tag_frame = f"tag25h9:{tag_id}"
        else:
            self.tag_frame = ""

    # -------------------- Service Helpers (UNTOUCHED) --------------------
    def set_mode(self, mode):
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.mode_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().mode_sent

    def arm(self, status):
        req = CommandBool.Request()
        req.value = status
        future = self.arm_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    def takeoff(self, altitude):
        req = CommandTOL.Request()
        req.altitude = float(altitude)
        future = self.takeoff_srv.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result().success

    # -------------------- AUTO TUNING METHODS (EXACT COPY + MINIMAL ADAPTATION) --------------------
    # Initialisation is 100% identical (blocking input for min/max gains)
    def Initialisation(self):
        self.get_logger().info('Enter the minimum Kp value for pitch, roll and z axis respectively')
        self.Kp_min[0] = float(input())
        self.Kp_min[1] = float(input())
        self.Kp_min[2] = float(input())
        self.get_logger().info(f'Kp_min = {self.Kp_min}')
        self.get_logger().info('Enter the maximum Kp value for pitch, roll and z axis respectively')
        self.Kp_max[0] = float(input())
        self.Kp_max[1] = float(input())
        self.Kp_max[2] = float(input())
        self.get_logger().info(f'Kp_max = {self.Kp_max}')
        self.get_logger().info('Enter the minimum Kd value for pitch, roll and z axis respectively')
        self.Kd_min[0] = float(input())
        self.Kd_min[1] = float(input())
        self.Kd_min[2] = float(input())
        self.get_logger().info(f'Kd_min = {self.Kd_min}')
        self.get_logger().info('Enter the maximum Kd value for pitch, roll and z axis respectively')
        self.Kd_max[0] = float(input())
        self.Kd_max[1] = float(input())
        self.Kd_max[2] = float(input())
        self.get_logger().info(f'Kd_max = {self.Kd_max}')
        self.get_logger().info('Enter the minimum Ki value for pitch, roll and z axis respectively')
        self.Ki_min[0] = float(input())
        self.Ki_min[1] = float(input())
        self.Ki_min[2] = float(input())
        self.get_logger().info(f'Ki_min = {self.Ki_min}')
        self.get_logger().info('Enter the maximum Ki value for pitch, roll and z axis respectively')
        self.Ki_max[0] = float(input())
        self.Ki_max[1] = float(input())
        self.Ki_max[2] = float(input())
        self.get_logger().info(f'Ki_max = {self.Ki_max}')
        self.kp_auto_x = self.Kp_min[0]
        self.kp_auto_y = self.Kp_min[1]
        self.kp_auto_z = self.Kp_max[2]
        self.ki_auto_x = self.Ki_min[0]
        self.ki_auto_y = self.Ki_min[1]
        self.ki_auto_z = self.Ki_min[2]
        self.kd_auto_x = self.Kd_min[0]
        self.kd_auto_y = self.Kd_min[1]
        self.kd_auto_z = self.Kd_min[2]

    # motion() adapted ONLY for AprilTag error input (core PID call and state machine untouched)
    def motion(self):
        if self.flag_initialize == 0:
            self.Kp = [self.kp_auto_x, self.kp_auto_y, self.kp_auto_z]
            self.Ki = [self.ki_auto_x, self.ki_auto_y, self.ki_auto_z]
            self.Kd = [self.kd_auto_x, self.kd_auto_y, self.kd_auto_z]
            self.err[self.a] = self.err_x
            self.err[self.a + 1] = self.err_y
            self.err[self.a + 2] = self.err_z
            self.flag_initialize = 1
            self.PID(self.err)
        else:
            self.err[self.a] = self.err_x
            self.err[self.a + 1] = self.err_y
            self.err[self.a + 2] = self.err_z
            self.tuner_err_x[self.w] = self.err_x
            self.tuner_err_y[self.w] = self.err_y
            self.tuner_err_z[self.w] = self.err_z
            self.PID(self.err)

    # PID() is 100% identical to original (only buffer names for range calculations updated to avoid conflict)
    def PID(self, err):
        self.now_time = time.time()
        delta_time = self.now_time - self.prev_time
        if delta_time > 0.023:
            for f in range(self.a, self.a + 3, 1):
                self.sumerr[f] = self.sumerr[f] + self.Ki[f % 3] * (err[f]) * delta_time
                self.sumerr1[f] = (err[f] - self.err_prior[f]) / delta_time
                self.out[f % 3] = self.Kp[f % 3] * err[f] + self.sumerr[f] + self.Kd[f % 3] * self.sumerr1[f]
                self.err_prior[f] = err[f]
            self.prev_time = self.now_time

        self.now_whycon = time.time()
        delta_whycon = self.now_whycon - self.prev_whycon
        if delta_whycon > 0.030:
            # Kp tuning
            if self.flag_Kp_start == 0:
                self.get_logger().info('Kp tuning initiated')
                if (math.fabs(err[self.a]) < 1.8 and math.fabs(err[self.a + 1]) > 1.8 and math.fabs(err[self.a + 2]) > 0.3):
                    self.r = 0
                    self.err_z_auto = err[self.a + 2]
                    self.Kp[1] = self.Kp[1] + 0.01
                    self.Kp[2] = self.Kp[2] - (self.err_z_auto - self.err_prior_z_auto) / (delta_whycon * 100)
                    self.err_prior_z_auto = self.err_z_auto
                elif (math.fabs(err[self.a]) < 1.8 and math.fabs(err[self.a + 1]) > 1.8 and math.fabs(err[self.a + 2]) < 0.3):
                    self.r = 0
                    self.Kp[1] = self.Kp[1] + 0.1
                elif (math.fabs(err[self.a]) > 1.8 and math.fabs(err[self.a + 1]) < 1.8 and math.fabs(err[self.a + 2]) > 0.3):
                    self.r = 0
                    self.err_z_auto = err[self.a + 2]
                    self.Kp[0] = self.Kp[0] + 0.1
                    self.Kp[2] = self.Kp[2] - (self.err_z_auto - self.err_prior_z_auto) / (delta_whycon * 100)
                    self.err_prior_z_auto = self.err_z_auto
                elif (math.fabs(err[self.a]) > 1.8 and math.fabs(err[self.a + 1]) < 1.8 and math.fabs(err[self.a + 2]) < 0.3):
                    self.r = 0
                    self.Kp[0] = self.Kp[0] + 0.1
                elif (math.fabs(err[self.a]) > 1.8 and math.fabs(err[self.a + 1]) > 1.8 and math.fabs(err[self.a + 2]) > 0.3):
                    self.r = 0
                    self.err_x_auto = err[self.a]
                    self.err_y_auto = err[self.a + 1]
                    self.err_z_auto = err[self.a + 2]
                    self.Kp[0] = self.Kp[0] - (math.fabs(self.err_x_auto) - math.fabs(self.err_prior_x_auto)) / (delta_whycon * 100)
                    self.Kp[1] = self.Kp[1] - (math.fabs(self.err_y_auto) - math.fabs(self.err_prior_y_auto)) / (delta_whycon * 100)
                    self.Kp[2] = self.Kp[2] - (self.err_z_auto - self.err_prior_z_auto) / (delta_whycon * 100)
                    self.err_prior_x_auto = self.err_x_auto
                    self.err_prior_y_auto = self.err_y_auto
                    self.err_prior_z_auto = self.err_z_auto
                elif (math.fabs(err[self.a]) > 1.8 and math.fabs(err[self.a + 1]) > 1.8 and math.fabs(err[self.a + 2]) < 0.3):
                    self.r = 0
                    self.err_x_auto = err[self.a]
                    self.err_y_auto = err[self.a + 1]
                    self.Kp[0] = self.Kp[0] - (math.fabs(self.err_x_auto) - math.fabs(self.err_prior_x_auto)) / (delta_whycon * 100)
                    self.Kp[1] = self.Kp[1] - (math.fabs(self.err_y_auto) - math.fabs(self.err_prior_y_auto)) / (delta_whycon * 100)
                    self.err_prior_x_auto = self.err_x_auto
                    self.err_prior_y_auto = self.err_y_auto
                elif (math.fabs(err[self.a]) <= 1.8 and math.fabs(err[self.a + 1]) <= 1.8 and math.fabs(err[self.a + 2]) > 0.3):
                    self.r = 0
                    self.err_z_auto = err[self.a + 2]
                    self.Kp[2] = self.Kp[2] - (self.err_z_auto - self.err_prior_z_auto) / (delta_whycon * 100)
                    self.err_prior_z_auto = self.err_z_auto
                elif (math.fabs(err[self.a]) <= 1.8 and math.fabs(err[self.a + 1]) <= 1.8 and math.fabs(err[self.a + 2]) <= 1.8):
                    self.r = self.r + 1
                    if self.r == 10:
                        self.flag_Kp_start = 1
                        self.r = 0
                self.get_logger().info(f'Kp: {self.Kp}')
                self.get_logger().info(f'Kd: {self.Kd}')
            # Kd tuning
            elif self.flag_Kp_start == 1:
                self.get_logger().info('Kp tuned')
                self.get_logger().info('Kd tuning initiated')
                range_x = math.fabs(max(self.tuner_err_x) - min(self.tuner_err_x))
                range_y = math.fabs(max(self.tuner_err_y) - min(self.tuner_err_y))
                range_z = math.fabs(max(self.tuner_err_z) - min(self.tuner_err_z))
                if (range_x <= 1 and range_y <= 1 and range_z <= 0.5):
                    self.r = self.r + 1
                    if self.r == 50:
                        self.r = 0
                        self.flag_Kp_start = 2
                elif (range_x <= 1 and range_y <= 1 and range_z > 0.5):
                    self.err_z_auto = err[self.a + 2]
                    self.Kd[2] = self.Kd[2] + (math.fabs((self.err_z_auto) - (self.err_prior_z_auto)) / (delta_whycon * 150))
                    self.err_prior_z_auto = self.err_z_auto
                elif (range_x < 1 and range_y > 1 and range_z < 0.5):
                    self.r = 0
                    self.err_y_auto = err[self.a + 1]
                    self.Kd[1] = self.Kd[1] + (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 150))
                    self.err_prior_y_auto = self.err_y_auto
                elif (range_x < 1 and range_y > 1 and range_z > 0.5):
                    self.r = 0
                    self.err_z_auto = err[self.a + 2]
                    self.err_y_auto = err[self.a + 1]
                    self.Kd[2] = self.Kd[2] + (math.fabs((self.err_z_auto - self.err_prior_z_auto)) / (delta_whycon * 150))
                    self.Kd[1] = self.Kd[1] + (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 150))
                    self.err_prior_z_auto = self.err_z_auto
                    self.err_prior_y_auto = self.err_y_auto
                elif (range_x > 1 and range_y < 1 and range_z < 0.5):
                    self.r = 0
                    self.err_x_auto = err[self.a]
                    self.Kd[0] = self.Kd[0] + (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 150))
                    self.err_prior_x_auto = self.err_x_auto
                elif (range_x > 1 and range_y < 1 and range_z > 0.5):
                    self.r = 0
                    self.err_x_auto = err[self.a]
                    self.err_z_auto = err[self.a + 2]
                    self.Kd[0] = self.Kd[0] + (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 150))
                    self.Kd[2] = self.Kd[2] + (math.fabs((self.err_z_auto - self.err_prior_z_auto)) / (delta_whycon * 150))
                    self.err_prior_z_auto = self.err_z_auto
                    self.err_prior_x_auto = self.err_x_auto
                elif (range_x > 1 and range_y > 1 and range_z < 0.5):
                    self.r = 0
                    self.err_x_auto = err[self.a]
                    self.err_y_auto = err[self.a + 1]
                    self.Kd[0] = self.Kd[0] + (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 100))
                    self.Kd[1] = self.Kd[1] + (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 100))
                    self.err_prior_x_auto = self.err_x_auto
                    self.err_prior_y_auto = self.err_y_auto
                elif (range_x > 1 and range_y > 1 and range_z > 0.5):
                    self.r = 0
                    self.err_x_auto = err[self.a]
                    self.err_y_auto = err[self.a + 1]
                    self.err_z_auto = err[self.a + 2]
                    self.Kd[0] = self.Kd[0] + (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 100))
                    self.Kd[1] = self.Kd[1] + (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 100))
                    self.Kd[2] = self.Kd[2] + (math.fabs(self.err_z_auto - self.err_prior_z_auto) / (delta_whycon * 100))
                    self.err_prior_x_auto = self.err_x_auto
                    self.err_prior_y_auto = self.err_y_auto
                    self.err_prior_z_auto = self.err_z_auto
                self.get_logger().info(f'Kp: {self.Kp}')
                self.get_logger().info(f'Kd: {self.Kd}')
            # Ki tuning
            elif self.flag_Kp_start == 2:
                self.get_logger().info('Kp tuned')
                self.get_logger().info('Kd tuning in progress')
                self.get_logger().info('Ki tuning initiated')
                self.now_ki = time.time()
                if (self.now_ki - self.prev_ki) >= 1:
                    if (math.fabs(err[self.a]) > 0.3 and math.fabs(err[self.a + 1]) > 0.3 and math.fabs(err[self.a + 2]) > 0.3):
                        self.r = 0
                        self.Ki[0] = self.Ki[0] + 0.25
                        self.Ki[1] = self.Ki[1] + 0.25
                        self.Ki[2] = self.Ki[2] + 0.25
                    elif (math.fabs(err[self.a]) > 0.3 and math.fabs(err[self.a + 1]) > 0.3 and math.fabs(err[self.a + 2]) < 0.3):
                        self.r = 0
                        self.Ki[0] = self.Ki[0] + 0.25
                        self.Ki[1] = self.Ki[1] + 0.25
                    elif (math.fabs(err[self.a]) < 0.3 and math.fabs(err[self.a + 1]) > 0.3 and math.fabs(err[self.a + 2]) > 0.3):
                        self.r = 0
                        self.Ki[1] = self.Ki[1] + 0.25
                        self.Ki[2] = self.Ki[2] + 0.25
                    elif (math.fabs(err[self.a]) < 0.3 and math.fabs(err[self.a + 1]) > 0.3 and math.fabs(err[self.a + 2]) < 0.3):
                        self.r = 0
                        self.Ki[1] = self.Ki[1] + 0.25
                    elif (math.fabs(err[self.a]) > 0.3 and math.fabs(err[self.a + 1]) < 0.3 and math.fabs(err[self.a + 2]) > 0.3):
                        self.r = 0
                        self.Ki[0] = self.Ki[0] + 0.25
                        self.Ki[2] = self.Ki[2] + 0.25
                    elif (math.fabs(err[self.a]) > 0.3 and math.fabs(err[self.a + 1]) < 0.3 and math.fabs(err[self.a + 2]) < 0.3):
                        self.r = 0
                        self.Ki[0] = self.Ki[0] + 0.25
                    elif (math.fabs(err[self.a]) <= 0.3 and math.fabs(err[self.a + 1]) <= 0.3 and math.fabs(err[self.a + 2]) > 0.3):
                        self.r = 0
                        self.Ki[2] = self.Ki[2] + 0.25
                    elif (math.fabs(err[self.a]) <= 0.3 and math.fabs(err[self.a + 1]) <= 0.3 and math.fabs(err[self.a + 2]) <= 0.3):
                        self.r = self.r + 1
                        if self.r == 10:
                            self.get_logger().info('Auto tuning completed. All good :)')
                            self.flag_Kp_start = 3
                    # Kd damping (exact original logic)
                    range_x_ki = math.fabs(max(self.tuner_err_x) - min(self.tuner_err_x))
                    range_y_ki = math.fabs(max(self.tuner_err_y) - min(self.tuner_err_y))
                    range_z_ki = math.fabs(max(self.tuner_err_z) - min(self.tuner_err_z))
                    if (range_x_ki <= 1 and range_y_ki <= 1 and range_z_ki > 0.7):
                        self.err_z_auto = err[self.a + 2]
                        self.Kd[2] = self.Kd[2] + (math.fabs((self.err_z_auto) - (self.err_prior_z_auto)) / (delta_whycon * 150))
                        self.err_prior_z_auto = self.err_z_auto
                    elif (range_x_ki < 1 and range_y_ki > 1 and range_y_ki <= 2 and range_z_ki < 0.7):
                        self.err_y_auto = err[self.a + 1]
                        self.Kd[1] = self.Kd[1] + (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 150))
                        self.err_prior_y_auto = self.err_y_auto
                    elif (range_x_ki < 1 and range_y_ki > 2 and range_z_ki < 0.7):
                        self.err_y_auto = err[self.a + 1]
                        self.Kd[1] = self.Kd[1] - (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 150))
                        self.err_prior_y_auto = self.err_y_auto
                    elif (range_x_ki < 1 and range_y_ki > 1 and range_y_ki <= 2 and range_z_ki > 0.7):
                        self.err_z_auto = err[self.a + 2]
                        self.err_y_auto = err[self.a + 1]
                        self.Kd[2] = self.Kd[2] + (math.fabs((self.err_z_auto - self.err_prior_z_auto)) / (delta_whycon * 150))
                        self.Kd[1] = self.Kd[1] + (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 150))
                        self.err_prior_z_auto = self.err_z_auto
                        self.err_prior_y_auto = self.err_y_auto
                    elif (range_x_ki < 1 and range_y_ki > 2 and range_z_ki > 0.7):
                        self.err_z_auto = err[self.a + 2]
                        self.err_y_auto = err[self.a + 1]
                        self.Kd[2] = self.Kd[2] + (math.fabs((self.err_z_auto - self.err_prior_z_auto)) / (delta_whycon * 150))
                        self.Kd[1] = self.Kd[1] - (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 150))
                        self.err_prior_z_auto = self.err_z_auto
                        self.err_prior_y_auto = self.err_y_auto
                    elif (range_x_ki > 1 and range_x_ki <= 2 and range_y_ki < 1 and range_z_ki < 0.7):
                        self.err_x_auto = err[self.a]
                        self.Kd[0] = self.Kd[0] + (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 150))
                        self.err_prior_x_auto = self.err_x_auto
                    elif (range_x_ki > 2 and range_y_ki < 1 and range_z_ki < 0.7):
                        self.err_x_auto = err[self.a]
                        self.Kd[0] = self.Kd[0] - (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 150))
                        self.err_prior_x_auto = self.err_x_auto
                    elif (range_x_ki > 1 and range_x_ki <= 2 and range_y_ki < 1 and range_z_ki > 0.7):
                        self.err_x_auto = err[self.a]
                        self.err_z_auto = err[self.a + 2]
                        self.Kd[0] = self.Kd[0] + (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 150))
                        self.Kd[2] = self.Kd[2] + (math.fabs((self.err_z_auto - self.err_prior_z_auto)) / (delta_whycon * 150))
                        self.err_prior_z_auto = self.err_z_auto
                        self.err_prior_x_auto = self.err_x_auto
                    elif (range_x_ki > 2 and range_y_ki < 1 and range_z_ki > 0.7):
                        self.err_x_auto = err[self.a]
                        self.err_z_auto = err[self.a + 2]
                        self.Kd[0] = self.Kd[0] - (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 150))
                        self.Kd[2] = self.Kd[2] + (math.fabs((self.err_z_auto - self.err_prior_z_auto)) / (delta_whycon * 150))
                        self.err_prior_z_auto = self.err_z_auto
                        self.err_prior_x_auto = self.err_x_auto
                    elif (range_x_ki > 1 and range_x_ki <= 2 and range_y_ki > 1 and range_y_ki <= 2 and range_z_ki < 0.7):
                        self.err_x_auto = err[self.a]
                        self.err_y_auto = err[self.a + 1]
                        self.Kd[0] = self.Kd[0] + (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 100))
                        self.Kd[1] = self.Kd[1] + (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 100))
                        self.err_prior_x_auto = self.err_x_auto
                        self.err_prior_y_auto = self.err_y_auto
                    elif (range_x_ki > 2 and range_y_ki > 2 and range_z_ki < 0.7):
                        self.err_x_auto = err[self.a]
                        self.err_y_auto = err[self.a + 1]
                        self.Kd[0] = self.Kd[0] - (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 100))
                        self.Kd[1] = self.Kd[1] - (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 100))
                        self.err_prior_x_auto = self.err_x_auto
                        self.err_prior_y_auto = self.err_y_auto
                    elif (range_x_ki > 1 and range_x_ki <= 2 and range_y_ki > 1 and range_y_ki <= 2 and range_z_ki > 0.7):
                        self.err_x_auto = err[self.a]
                        self.err_y_auto = err[self.a + 1]
                        self.err_z_auto = err[self.a + 2]
                        self.Kd[0] = self.Kd[0] + (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 100))
                        self.Kd[1] = self.Kd[1] + (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 100))
                        self.Kd[2] = self.Kd[2] + (math.fabs(self.err_z_auto - self.err_prior_z_auto) / (delta_whycon * 100))
                        self.err_prior_x_auto = self.err_x_auto
                        self.err_prior_y_auto = self.err_y_auto
                        self.err_prior_z_auto = self.err_z_auto
                    elif (range_x_ki > 2 and range_y_ki > 2 and range_z_ki > 0.7):
                        self.err_x_auto = err[self.a]
                        self.err_y_auto = err[self.a + 1]
                        self.err_z_auto = err[self.a + 2]
                        self.Kd[0] = self.Kd[0] - (math.fabs(self.err_x_auto - self.err_prior_x_auto) / (delta_whycon * 100))
                        self.Kd[1] = self.Kd[1] - (math.fabs(self.err_y_auto - self.err_prior_y_auto) / (delta_whycon * 100))
                        self.Kd[2] = self.Kd[2] + (math.fabs(self.err_z_auto - self.err_prior_z_auto) / (delta_whycon * 100))
                        self.err_prior_x_auto = self.err_x_auto
                        self.err_prior_y_auto = self.err_y_auto
                        self.err_prior_z_auto = self.err_z_auto
                    self.prev_ki = self.now_ki
                    self.get_logger().info(f'Kp: {self.Kp}')
                    self.get_logger().info(f'Kd: {self.Kd}')
                    self.get_logger().info(f'Ki: {self.Ki}')
            # PID capping (exact)
            if self.Kp[0] <= self.Kp_min[0]:
                self.Kp[0] = self.Kp_min[0]
            elif self.Kp[0] >= self.Kp_max[0]:
                self.Kp[0] = self.Kp_max[0]
            if self.Kp[1] <= self.Kp_min[1]:
                self.Kp[1] = self.Kp_min[1]
            elif self.Kp[1] >= self.Kp_max[1]:
                self.Kp[1] = self.Kp_max[1]
            if self.Kp[2] <= self.Kp_min[2]:
                self.Kp[2] = self.Kp_min[2]
            elif self.Kp[2] >= self.Kp_max[2]:
                self.Kp[2] = self.Kp_max[2]
            if self.Kd[0] >= self.Kd_max[0]:
                self.Kd[0] = self.Kd_max[0]
            elif self.Kd[0] <= self.Kd_min[0]:
                self.Kd[0] = self.Kd_min[0]
            if self.Kd[1] >= self.Kd_max[1]:
                self.Kd[1] = self.Kd_max[1]
            elif self.Kd[1] <= self.Kd_min[1]:
                self.Kd[1] = self.Kd_min[1]
            if self.Kd[2] >= self.Kd_max[2]:
                self.Kd[2] = self.Kd_max[2]
            elif self.Kd[2] <= self.Kd_min[2]:
                self.Kd[2] = self.Kd_min[2]
            if self.Ki[0] >= self.Ki_max[0]:
                self.Ki[0] = self.Ki_max[0]
            elif self.Ki[0] <= self.Ki_min[0]:
                self.Ki[0] = self.Ki_min[0]
            if self.Ki[1] >= self.Ki_max[1]:
                self.Ki[1] = self.Ki_max[1]
            elif self.Ki[1] <= self.Ki_min[1]:
                self.Ki[1] = self.Ki_min[1]
            if self.Ki[2] >= self.Ki_max[2]:
                self.Ki[2] = self.Ki_max[2]
            elif self.Ki[2] <= self.Ki_min[2]:
                self.Ki[2] = self.Ki_min[2]
            # index management (exact)
            if self.a == 15:
                self.a = 0
            else:
                self.a = self.a + 3
            if self.w == 53:
                self.w = 0
            else:
                self.w = self.w + 1
            self.prev_whycon = self.now_whycon

    # -------------------- Main Mission Logic (UNTOUCHED) --------------------
    def run_mission(self):
        self.get_logger().info("Waiting for mission initiation...")
        while rclpy.ok() and not self.mission_initiated:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Mission Started!")

        self.get_logger().info("Waiting for FCU connection...")
        while not self.state.connected:
            rclpy.spin_once(self, timeout_sec=0.1)

        while self.local_pos.header.stamp.sec == 0:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.initial_alt = self.local_pos.pose.position.z

        self.get_logger().info("Setting GUIDED mode and Arming...")
        self.set_mode("GUIDED")
        self.arm(True)
        time.sleep(2)

        target_alt_rel = self.get_parameter("takeoff_alt").get_parameter_value().double_value
        self.get_logger().info(f"Taking off to {target_alt_rel}m")
        self.takeoff(target_alt_rel)

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.local_pos.pose.position.z >= (self.initial_alt + target_alt_rel - 0.3):
                break

        self.get_logger().info("Hovering to search for tag...")
        time.sleep(3)

        self.precision_landing_loop()

    # -------------------- Precision Landing (core sequence UNTOUCHED, tuning inserted before descent) --------------------
    def precision_landing_loop(self):
        self.get_logger().info("Starting Precision Landing")

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)

            found_now = False

            if self.tag_frame:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.camera_frame,
                        self.tag_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.02)
                    )

                    self.err_x = trans.transform.translation.x
                    self.err_y = trans.transform.translation.y
                    self.err_z = trans.transform.translation.z
                    found_now = True

                except (LookupException, ConnectivityException, ExtrapolationException):
                    found_now = False

            vel = TwistStamped()
            vel.header.stamp = self.get_clock().now().to_msg()
            vel.header.frame_id = "base_link"

            if found_now:
                if not self.tuning_started:
                    self.get_logger().info("Tag found. Starting auto PID tuning for X and Y (Z untouched)...")
                    try:
                        self.Initialisation()
                    except Exception as e:
                        self.get_logger().error(f'Initialization error: {e}')
                    self.tuning_started = True
                    self.flag_initialize = 0
                    self.flag_Kp_start = 0
                    self.r = 0
                    self.start_time_tuning = time.time()
                    self.get_logger().info('Auto tuning initiated (exact original logic).')

                if not self.tuning_complete:
                    self.motion()
                    # clamp (exact original)
                    if self.out[0] > self.max_vel:
                        self.out[0] = self.max_vel
                    elif self.out[0] < self.min_vel:
                        self.out[0] = self.min_vel
                    if self.out[1] > self.max_vel:
                        self.out[1] = self.max_vel
                    elif self.out[1] < self.min_vel:
                        self.out[1] = self.min_vel
                    # X/Y only - map with original landing sign convention
                    vx = -self.out[0]
                    vy = self.out[1]
                    vz = 0.0

                    # stable check (exact same tolerance as original waypoint logic)
                    if self.flag_Kp_start == 3:
                        within_tol = (math.fabs(self.err_x) <= 0.4 and
                                      math.fabs(self.err_y) <= 0.4 and
                                      math.fabs(self.err_z) <= 0.4)
                        current_time = time.time()
                        if within_tol:
                            if self.stable_start_time is None:
                                self.stable_start_time = current_time
                                self.get_logger().info('Entered stable zone, starting 3s timer.')
                            stable_duration = current_time - self.stable_start_time
                            if stable_duration >= 3.0:
                                self.get_logger().info('Auto tuning completed. Continuing precision landing descent with tuned PID.')
                                self.tuning_complete = True
                                self.stable_start_time = None
                        else:
                            if self.stable_start_time is not None:
                                self.get_logger().info('Exited stable zone, resetting timer.')
                            self.stable_start_time = None
                else:
                    # continue using exact tuned PID for X/Y
                    self.motion()
                    if self.out[0] > self.max_vel:
                        self.out[0] = self.max_vel
                    elif self.out[0] < self.min_vel:
                        self.out[0] = self.min_vel
                    if self.out[1] > self.max_vel:
                        self.out[1] = self.max_vel
                    elif self.out[1] < self.min_vel:
                        self.out[1] = self.min_vel
                    vx = -self.out[0]
                    vy = self.out[1]
                    error_mag = np.sqrt(self.err_x**2 + self.err_y**2)
                    vz = -0.15 if error_mag < 0.1 else 0.0
            else:
                vx = 0.0
                vy = 0.0
                vz = 0.0

            # publish velocity (core landing logic untouched)
            max_v = self.get_parameter("max_vel_xy").value
            vel.twist.linear.x = float(np.clip(vx, -max_v, max_v))
            vel.twist.linear.y = float(np.clip(vy, -max_v, max_v))
            vel.twist.linear.z = float(vz)
            self.vel_pub.publish(vel)

            # Landing condition (100% original)
            alt_rel = self.local_pos.pose.position.z - self.initial_alt
            if alt_rel < self.get_parameter("land_alt_threshold").value:
                self.get_logger().info("Switching to LAND mode")
                self.set_mode("LAND")
                break


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousPrecisionLanding()

    try:
        node.run_mission()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()