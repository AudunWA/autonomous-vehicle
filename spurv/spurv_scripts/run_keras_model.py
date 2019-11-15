#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import os

"""
This script allows one to select and run a Keras model (in h5 format), stored in the "models" subfolder.
"""

import subprocess
import imp
import cv2
import numpy as np
import threading
import datetime

"""
Xbox controller input IDs
"""
DISABLE_AUTONOMOUS = 4  # =LB-button on xbox
ENABLE_AUTONOMOUS = 5  # =RB-button on xbox
LEFT_HLC_BTN = 2  # X
FORWARD_HLC_BTN = 0  # A
RIGHT_HLC_BTN = 1  # B
CHANGE_MODEL_AXIS = 6  # D-pad left/right


JOY_THROTTLE_SECONDS = 1
SELECT_MODEL_THROTTLE_SECONDS = 0.2


"""
Hardcoded model parameters
"""
SEQUENCE_SPACE = 3
IMAGE_INTERVAL = 60

MODELS_FOLDER = "models"


def get_model_list():
    from os import listdir
    from os.path import isfile, join
    return [f for f in listdir(MODELS_FOLDER) if isfile(join(MODELS_FOLDER, f))]


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def steering_loss(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


class RunModel(object):

    def __init__(self):
        self.autonomous_mode = False
        self.image_buffer = []
        self.cv_bridge = CvBridge()

        # Model variables
        self.models = get_model_list()
        self.model = None
        self.model_index = 0
        self.next_model_index = 0
        self.img_shape = None
        self.sequence_length = None
        self.sine_steering = None

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
        self.current_hlc = 1
        rospy.loginfo('Spurv autonomous driver initialized')

    def init_model(self, index):
        self.autonomous_mode = False
        self.model = None
        self.graph = None
        self.image_history = []
        self.hlc_history = []
        self.info_history = []
        self.current_hlc = 1

        print("Autonomous mode disabled.")
        K.clear_session()

        if index >= len(self.models):
            print("Invalid model index: " + str(index))
            return

        print("Initializing model " + self.models[index])
        # print(self.model.summary())

        self.model = load_model(MODELS_FOLDER + "/" + self.models[index], custom_objects={'custom': steering_loss})
        self.model_index = index
        self.next_model_index = index
        self.graph = tf.get_default_graph()
        print("Setting graph to " + str(self.graph))

        (self.img_shape, self.sequence_length, self.sine_steering) = self.get_model_params(self.model)

        print("Loaded model " + self.models[index])
        print("Image shape: " + str(self.img_shape) + ", sequence length: " + str(
            self.sequence_length) + ", sine steering? " + str(self.sine_steering))

    """
    Extracts some info about the model from its input and output layers
    """

    def get_model_params(self, model):
        forward_image_input_layer = self.model.get_layer('forward_image_input')
        steer_pred_output_layer = self.model.get_layer('steer_pred')

        (_, sequence_length, height, width, channels) = forward_image_input_layer.input_shape
        sine_steering = (steer_pred_output_layer.output_shape == (None, 10))
        return (height, width, channels), sequence_length, sine_steering

    def init_pub_sub(self):
        self.image_subscriber = rospy.Subscriber('/fwd_camera/image_raw'
                                                 , Image, self.on_image_callback)
        self.joystick_subscriber = rospy.Subscriber('/joy', Joy,
                                                    self.on_joy_callback)
        self.steering_publisher = rospy.Publisher('/ackermann_cmd',
                                                  AckermannDriveStamped, queue_size=1)

    def store_image(self, image):
        self.image_buffer.insert(0, image)
        if len(self.image_buffer) > 2 * IMAGE_INTERVAL:
            self.image_buffer.pop(-1)

    def sin_decoder(self, x, angle):
        return np.sin(((2 * np.pi * (x - 1)) / 9) - ((angle * np.pi) / 2))

    def get_prediction(self, image):

        image = cv2.resize(image, (self.img_shape[1], self.img_shape[0]))

        # Normalize and add image to history
        self.image_history.append(image / 255.0 - 0.5)

        hlc = [0, 0, 0]
        hlc[self.current_hlc] = 1
        self.hlc_history.append(hlc)

        req_len = (self.sequence_length - 1) * (SEQUENCE_SPACE + 1) + 1
        if len(self.image_history) > req_len:
            self.hlc_history.pop(0)
            self.image_history.pop(0)

        if len(self.image_history) < req_len:
            return 0, 0

        image_sequence = np.array([self.image_history[0::SEQUENCE_SPACE + 1]])
        hlc_sequence = np.array([self.hlc_history[0::SEQUENCE_SPACE + 1]])

        with self.graph.as_default():

            prediction = \
                self.model.predict({'forward_image_input': image_sequence,
                                    'hlc_input': hlc_sequence})

            steer = prediction[0][0]
            throttle = prediction[1][0][0]

            if self.sine_steering:
                steer_curve_parameters = curve_fit(self.sin_decoder, np.arange(1, 11, 1), steer)[0]
                steer_angle = steer_curve_parameters[0]
            else:
                steer_angle = steer[0]

            return throttle, steer_angle

    def on_image_callback(self, data):
        if not self.autonomous_mode:
            return

        image = self.cv_bridge.imgmsg_to_cv2(data, 'bgr8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        (throttle, angle) = self.get_prediction(image)

        self.publish_steering(throttle, angle)

    def on_joy_callback(self, joyMessage):
        enable = joyMessage.buttons[ENABLE_AUTONOMOUS]
        disable = joyMessage.buttons[DISABLE_AUTONOMOUS]
        if not self.is_joy_callback_throttled:
            if bool(enable):
                if self.next_model_index != self.model_index:
                    self.change_model()
                else:
                    self.autonomous_mode = True
                    print('Autonomous mode enabled')
            if bool(disable):
                self.autonomous_mode = False
                print('Autonomous mode disabled')

            left_hlc = joyMessage.buttons[LEFT_HLC_BTN]
            forward_hlc = joyMessage.buttons[FORWARD_HLC_BTN]
            right_hlc = joyMessage.buttons[RIGHT_HLC_BTN]

            if bool(left_hlc):
                self.current_hlc = 0
            elif bool(forward_hlc):
                self.current_hlc = 1
            elif bool(right_hlc):
                self.current_hlc = 2

            if self.joy_throttle_timer:
                self.joy_throttle_timer.cancel()

            self.joy_throttle_timer = threading.Timer(JOY_THROTTLE_SECONDS, self.remove_joy_throttle)
            self.joy_throttle_timer.start()
            self.is_joy_callback_throttled = True

        if self.is_model_callback_throttled:
            return

        self.set_or_reset_next_model_timer(SELECT_MODEL_THROTTLE_SECONDS, self.remove_model_throttle)
        self.is_model_callback_throttled = True

        # Go to next or previous model if requested
        change_model_axis = joyMessage.axes[CHANGE_MODEL_AXIS]
        if abs(change_model_axis) != 0.0:
            self.next_model_index = (self.next_model_index + int(change_model_axis)) % len(self.models)
            print("Model to load: " + self.models[self.next_model_index])

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
        imp.find_module('rospy')
        return True
    except ImportError:
        return False


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, shell=True, cwd=os.path.dirname(os.path.realpath(__file__)))
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
        import rospy
        from sensor_msgs.msg import Joy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge, CvBridgeError
        from ackermann_msgs.msg import AckermannDriveStamped

        """
           Solves a memory issue, needs to be done before importing Keras 
        """
        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        session = tf.Session(config=config)
        from tensorflow.keras.models import load_model, Model
        import tensorflow.keras.backend as K
        from scipy.optimize import curve_fit

        frame_id = rospy.get_param('~frame_id', 'odom')
        max_accel_x = rospy.get_param('~acc_lim_x', 1.0)
        max_jerk_x = rospy.get_param('~jerk_lim_x', 0.0)

        rospy.init_node('autnomous_driver')
        driver = RunModel()
        try:
            rospy.spin()
        except KeyboardInterrupt, interrupt:
            pass
    else:
        print("Running in plain Python, starting with rosrun...")
        for path in execute(["/bin/bash -i -c 'rosrun spurv_examples run_keras_model.py'"]):
            print(path, end="")
