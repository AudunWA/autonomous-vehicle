#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import time

import math
import os

from model_helpers import get_image_array

"""
This script allows one to select and run a Keras model (in h5 format), stored in the "models" subfolder.
"""

import subprocess
import imp
import cv2
import numpy as np
import threading

"""
Xbox controller input IDs
"""
DISABLE_AUTONOMOUS = 4  # =LB-button on xbox
ENABLE_AUTONOMOUS = 5  # =RB-button on xbox
LEFT_HLC_BTN = 2  # X
FOLLOW_LANE_HLC_BTN = 0  # A
STRAIGHT_HLC_BTN = 3  # Y
RIGHT_HLC_BTN = 1  # B
CHANGE_MODEL_AXIS = 6  # D-pad left/right

JOY_THROTTLE_SECONDS = 0.2
SELECT_MODEL_THROTTLE_SECONDS = 0.2

"""
Hardcoded model parameters
"""
SEQUENCE_SPACE = 3
IMAGE_INTERVAL = 60

MODELS_GLOB = "/media/nvidia/NTNUspurvSD/audun/master_models/*/*.h5"

# HLCs
HLC_LEFT = 0
HLC_RIGHT = 1
HLC_STRAIGHT = 2
HLC_LANEFOLLOW = 3


def get_model_list():
    print("Looking for models in " + MODELS_GLOB)
    from os import listdir
    from os.path import isfile, join
    from glob import glob
    return sorted([f for f in glob(MODELS_GLOB)])


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def steering_loss(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


def get_model_params(model):
    """
    Extracts some info about the model from its input and output layers
    """

    forward_image_input_layer = model.get_layer('forward_image_input')
    steer_pred_output_layer = model.get_layer('steer_pred')

    (_, sequence_length, height, width, channels) = forward_image_input_layer.input_shape
    sine_steering = (steer_pred_output_layer.output_shape == (None, 10))
    return (height, width, channels), sequence_length, sine_steering


class RunModel(object):

    def __init__(self):
        self.autonomous_mode = False
        self.cv_bridge = CvBridge()

        # Model variables
        self.models = get_model_list()
        self.model = None
        self.is_loading_model = False
        self.model_index = -1
        self.next_model_index = 0
        self.sequence_length = None
        self.sine_steering = None

        self.img_shape = None
        self.height = None
        self.width = None
        self.channels = None

        # Tensorflow graph
        self.graph = None

        self.next_model_timer = None
        self.joy_throttle_timer = None
        self.is_joy_callback_throttled = False
        self.is_model_callback_throttled = False

        self.init_pub_sub()
        self.image_history = []
        self.hlc_history = []
        self.info_history = []
        self.current_hlc = HLC_LANEFOLLOW

        self.odometry_speed = 0

        # Measure inference speed
        self.start_time = time.time()
        self.inference_counter = 0

        rospy.loginfo('Spurv autonomous driver initialized')

    def init_model(self, index):
        self.is_loading_model = True
        self.autonomous_mode = False
        self.model = None
        self.graph = None
        self.image_history = []
        self.hlc_history = []
        self.info_history = []
        self.current_hlc = HLC_LANEFOLLOW

        print("Autonomous mode disabled.")
        K.clear_session()

        if index >= len(self.models):
            print("Invalid model index: " + str(index))
            return

        model_name = self.models[index]
        print("Initializing model " + model_name)
        # print(self.model.summary())

        activation = "relu" # "relu" if any([prefix in model_name for prefix in ["89"]]) else "elu"
        print("Activation: " + activation)
        K.clear_session()
        freeze = "no_freeze" not in model_name
        if freeze: #"depth" in self.models[index]:
            print("Found seg+depth model")
            from model_structures import mobilenet_segm_depth
            self.model = mobilenet_segm_depth.get_mobilenet_segm_depth(1, False, freeze, activation)
            self.model.load_weights(model_name)
        else:
            print("Found rgb model (no freeze)")
            from model_structures import mobilenet_segm
            self.model = mobilenet_segm.get_mobilenet_segm(1, False, False, activation)
            self.model.load_weights(model_name)
        # elif "concat_hlc" in self.models[index]:
        #     print("Found concat hlc model")
        #     from model_structures import target_speed_concat_hlc
        #     self.model = target_speed_concat_hlc.get_target_speed_concat_hlc(5, True, freeze)
        #     self.model.load_weights(self.models[index])
        # elif  "act10" in self.models[index]:
        #     print("Found act10 model")
        #     from model_structures.target_speed_act10 import get_target_speed_act10
        #     self.model = get_target_speed_act10(1, True, freeze)
        #     self.model.load_weights(self.models[index])
        # else:
        #     print("Found generic model")
        #     self.model = load_model(self.models[index], compile=False)
        self.model_index = index
        self.next_model_index = index
        self.graph = tf.get_default_graph()
        print("Setting graph to " + str(self.graph))

        (self.height, self.width, self.channels), self.sequence_length, self.sine_steering = get_model_params(self.model)
        self.img_shape = (self.height, self.width, self.channels)
        print("Loaded model " + model_name)
        print("Image shape: " + str(self.img_shape) + ", sequence length: " + str(
            self.sequence_length) + ", sine steering? " + str(self.sine_steering))
        self.is_loading_model = False

    def init_pub_sub(self):
        self.image_subscriber = rospy.Subscriber('/fwd_camera/image_raw'
                                                 , Image, self.on_image_callback)
        self.joystick_subscriber = rospy.Subscriber('/joy', Joy,
                                                    self.on_joy_callback)
        self.steering_publisher = rospy.Publisher('/ackermann_cmd',
                                                  AckermannDriveStamped, queue_size=1)
        self.odometry_subscriber = rospy.Subscriber("/odom", Odometry, self.on_odometry_callback)

    def sin_decoder(self, x, angle):
        return np.sin(((2 * np.pi * (x - 1)) / 9) - ((angle * np.pi) / 2))

    def get_prediction(self, image):
        # Normalize and add image to history
        self.image_history.append(get_image_array(image, width=self.width, height=self.height, imgNorm="sub_mean", ordering="channels_last"))

        # LEFT, RIGHT, STRAIGHT, LANEFOLLOW
        hlc = [0, 0, 0, 0]
        hlc[self.current_hlc] = 1
        self.hlc_history.append(np.array(hlc))
        print(hlc)

        CONSTANT_SPEED_LIMIT = 1 / 3.6
        CONSTANT_IS_GREEN_TRAFFIC_LIGHT = 1
        info = [
            float(self.odometry_speed * 3.6 / 30 - 1),
            float(CONSTANT_SPEED_LIMIT * 3.6 / 30 - 1),
            CONSTANT_IS_GREEN_TRAFFIC_LIGHT
        ]
        self.info_history.append(np.array(info))

        req_len = (self.sequence_length - 1) * (SEQUENCE_SPACE + 1) + 1
        if len(self.image_history) > req_len:
            self.hlc_history.pop(0)
            self.info_history.pop(0)
            self.image_history.pop(0)

        if len(self.image_history) < req_len:
            return 0, 0

        image_sequence = np.array([self.image_history[0::SEQUENCE_SPACE + 1]])
        hlc_sequence = np.array([self.hlc_history[0::SEQUENCE_SPACE + 1]])
        info_sequence = np.array([self.info_history[0::SEQUENCE_SPACE + 1]])

        with self.graph.as_default():

            input = {
                'forward_image_input': image_sequence,
                'hlc_input': hlc_sequence,
                "info_input": info_sequence
            }
            prediction = \
                self.model.predict(input)

            steer = prediction[0][0]
            
            # Target speed (in m/s)
            target_speed = prediction[1][0][0] * 100 / 3.6

            # Limit to 3.2 km/h for safety
            print("Wanted target speed " + str(target_speed * 3.6) + " km/h")
            target_speed = min(3.2 / 3.6, target_speed)

            # TEMP INDOOR TEST
            # target_speed = 0

            # Hack to boost min speed to get the robot driving
            target_speed = max(0.3, target_speed)

            if self.sine_steering:
                steer_curve_parameters = curve_fit(self.sin_decoder, np.arange(1, 11, 1), steer)[0]
                steer_angle = steer_curve_parameters[0]
            else:
                steer_angle = steer[0] * 2 - 1

            # Negate steer angle as it is opposite of CARLA
            steer_angle = -steer_angle

            # Boosting for dramatic effect
            steer_angle *= 2

            # Calculate and print inference speed
            self.inference_counter += 1
            if time.time() - self.start_time > 1:
                print("Inferences per second: " + str(self.inference_counter / (time.time() - self.start_time)))
                self.inference_counter = 0
                self.start_time = time.time()

            return target_speed, steer_angle



    def on_image_callback(self, data):
        if not self.autonomous_mode:
            return

        image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        (throttle, angle) = self.get_prediction(image)

        self.publish_steering(throttle, angle)

    def on_odometry_callback(self, data):
        linear_velocity = data.twist.twist.linear
        self.odometry_speed = math.sqrt(linear_velocity.x**2+linear_velocity.y**2+linear_velocity.z**2)
        # print("Speed" + str(self.odometry_speed))

    def on_joy_callback(self, joyMessage):
        enable = joyMessage.buttons[ENABLE_AUTONOMOUS]
        disable = joyMessage.buttons[DISABLE_AUTONOMOUS]
        left_hlc = joyMessage.buttons[LEFT_HLC_BTN]
        straight_hlc = joyMessage.buttons[STRAIGHT_HLC_BTN]
        follow_lane_hlc = joyMessage.buttons[FOLLOW_LANE_HLC_BTN]
        right_hlc = joyMessage.buttons[RIGHT_HLC_BTN]

        if not self.is_joy_callback_throttled:
            self.set_autonomous_mode(disable, enable)

            if bool(left_hlc):
                self.current_hlc = HLC_LEFT
                print("HLC: left")
            elif bool(straight_hlc):
                self.current_hlc = HLC_STRAIGHT
                print("HLC: forward")
            elif bool(right_hlc):
                self.current_hlc = HLC_RIGHT
                print("HLC: right")
            elif bool(follow_lane_hlc):
                self.current_hlc = HLC_LANEFOLLOW
                print("HLC: follow lane")

            if self.joy_throttle_timer:
                self.joy_throttle_timer.cancel()

            self.joy_throttle_timer = threading.Timer(JOY_THROTTLE_SECONDS, self.remove_joy_throttle)
            self.joy_throttle_timer.start()
            self.is_joy_callback_throttled = True

        if self.is_model_callback_throttled or self.is_loading_model:
            return

        self.set_or_reset_next_model_timer(SELECT_MODEL_THROTTLE_SECONDS, self.remove_model_throttle)
        self.is_model_callback_throttled = True

        # Go to next or previous model if requested
        change_model_axis = joyMessage.axes[CHANGE_MODEL_AXIS]
        if abs(change_model_axis) != 0.0:
            self.next_model_index = (self.next_model_index - int(change_model_axis)) % len(self.models)
            print("Model to load: " + self.models[self.next_model_index])

    def set_autonomous_mode(self, disable, enable):
        if bool(enable):
            if not self.is_loading_model and self.next_model_index != self.model_index:
                self.change_model()
            elif self.model is not None:
                self.autonomous_mode = True
                print('Autonomous mode enabled')
            else:
                print('Waiting for model to load...')
        if bool(disable):
            self.autonomous_mode = False
            print('Autonomous mode disabled')

    def remove_joy_throttle(self):
        self.is_joy_callback_throttled = False

    def remove_model_throttle(self):
        self.is_model_callback_throttled = False

    def publish_steering(self, throttle, angle):
        if self.autonomous_mode:
            print("Angle: " + str(angle) + ", throttle: " + str(throttle) + " m/s")
            msg = AckermannDriveStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = frame_id
            msg.drive.steering_angle = angle
            msg.drive.speed = throttle
            msg.drive.acceleration = max_accel_x
            msg.drive.jerk = max_jerk_x
            self.steering_publisher.publish(msg)

    def set_or_reset_next_model_timer(self, timeout, fn):
        if self.next_model_timer:
            self.next_model_timer.cancel()

        self.next_model_timer = threading.Timer(timeout, fn)
        self.next_model_timer.start()

    def change_model(self):
        process_thread = threading.Thread(target=self.init_model, args=(self.next_model_index,))
        process_thread.start()


def is_running_on_ros():
    try:
        import rospy
        rospy.get_param('~frame_id', 'odom')
        return True
    except:
        return False


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True,
                             cwd=os.path.dirname(os.path.realpath(__file__)))
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


if __name__ == '__main__':
    print("Starting script")
    if is_running_on_ros():
        print("Running on ROS!")

        # Your actual code here
        # Import ROS-dependencies here
        import rospy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Joy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        from ackermann_msgs.msg import AckermannDriveStamped

        """
           Solves a memory issue, needs to be done before importing Keras 
        """
        # import tensorflow as tf
        from keras.backend import tf

        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        from keras.models import load_model
        import keras.backend as K
        from scipy.optimize import curve_fit

        frame_id = rospy.get_param('~frame_id', 'odom')
        max_accel_x = rospy.get_param('~acc_lim_x', 1.0)
        max_jerk_x = rospy.get_param('~jerk_lim_x', 0.0)

        rospy.init_node('autonomous_driver')
        driver = RunModel()
        try:
            rospy.spin()
        except KeyboardInterrupt, interrupt:
            pass
    else:
        print("Running in plain Python, starting in bash...")
        for path in execute(["/bin/bash -i -c 'python " + __file__ + "'"]):
            print(path, end="")
