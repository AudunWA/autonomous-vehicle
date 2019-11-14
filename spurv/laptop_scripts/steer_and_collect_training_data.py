#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import errno
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import csv
import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import CompressedImage
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Joy
from datetime import datetime
from threading import Timer, Thread
import random
from time import sleep
from expiringdict import ExpiringDict

cache = ExpiringDict(max_len=1000, max_age_seconds=2)

RECORD_BUTTON = 7  # Start button
LEFT_HLC_BUTTON = 2  # X
FORWARD_HLC_BUTTON = 0  # A
RIGHT_HLC_BUTTON = 1  # B
TOGGLE_NOISE_BUTTON = 6  # BACK

HLC_RESET_SECONDS = 10
JOY_THROTTLE_SECONDS = 1
NOISE_DURATION_SECONDS = 1

HLC_LEFT = 0
HLC_FORWARD = 1
HLC_RIGHT = 2

IMAGE_SCALE = 0.3

"""
For steering
"""
SPEED_AXIS = 3
SERVO_AXIS = 0
LIGHTS_BUTTON = 2
MAX_SPEED = 0.6 # m/s

# Steering noise
MIN_NOISE = 0.2
MAX_NOISE = 0.6

frame_id = rospy.get_param("~frame_id", "odom")
max_accel_x = rospy.get_param("~acc_lim_x", 1.0)
max_jerk_x = rospy.get_param("~jerk_lim_x", 0.0)


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except OSError, exception:
        if exception.errno != errno.EEXIST:
            raise


class DataCollector:
    speedAxisValue = 0.0
    angleAxisValue = 0.0

    STORAGE_DIR = os.path.expanduser('~') + '/data_dump/'

    bridge = CvBridge()
    data_writer = None
    csv_file = None
    csv_field_names = ['speed', 'angle', 'high_level_command',
                       'image_path', 'ackerman_timestamp', 'img_timestamp', 'angle_w_noise']
    folder_name = None
    joystick_subscriber = None
    angle_publisher = None
    hlc_timer = None
    joy_throttle_timer = None
    is_joy_callback_throttled = False

    # Steering noise
    noise_enabled = False
    current_noise_angle = 0
    current_noise_peak = 0
    noise_add = 0
    current_noise = 0
    current_angle_with_noise = 0

    def __init__(self):
        print 'Initializing datacollector.'
        self.current_speed = 0
        self.current_angle = 0
        self.current_ackermann_timestamp = 0
        self.recording = False

        self.setup_collection_folders()

        self.csv_file = open(self.folder_name + 'data.csv', 'w+')
        self.data_writer = csv.writer(self.csv_file)

        self.data_writer.writerow(self.csv_field_names)

        self.high_level_command = HLC_FORWARD
        self.frame_seq_number = 0
        self.bridge = CvBridge()

        self.joystick_subscriber = rospy.Subscriber('/joy', Joy,
                                                    self._joy_callback)

        self._initPubSub()
        print("Spurv Joystick Driver initialized")
        self._initThd()

    def _initThd(self):
        """
        Initializes and starts the threads
        """
        self.thd = Thread(target=self._thdLoop)
        self.thd.daemon = True
        self.thd.start()

    def _initPubSub(self):
        """
        Initialize publishers and subscribers
        """
        self.ackermannPub = rospy.Publisher("/ackermann_cmd", AckermannDriveStamped, queue_size=1)

    def setup_collection_folders(self):
        self.folder_name = self.STORAGE_DIR \
                           + datetime.now().replace(microsecond=0).isoformat() + '/'

        safe_makedirs(self.folder_name)
        safe_makedirs(self.folder_name + 'images/')

    def start_collecting(self):

        print 'Starting data collection. Listening for commands and camera.'

        self.recording = True
        self.image_subscription = rospy.Subscriber('/fwd_camera/image_raw/compressed',
                                                   CompressedImage, self._image_callback)

    def stop_collecting(self):
        print 'Stopping data collection. Press Start to continue collecting.'
        self.recording = False
        self.image_subscription.unregister()

    def get_image_path(self):
        return self.folder_name + 'images/' \
               + str(self.frame_seq_number) + '.jpg'

    def _image_callback(self, data_loc):

        if not self.recording:
            return

        img_to_save = self.bridge.compressed_imgmsg_to_cv2(data_loc,
                                                           'bgr8')
        img_to_save = cv2.resize(img_to_save, (0, 0), fx=IMAGE_SCALE,
                                 fy=IMAGE_SCALE)

        img_path = self.get_image_path()
        img_timestamp = data_loc.header.stamp
        self.data_writer.writerow([self.current_speed,
                                   self.current_angle,
                                   self.high_level_command, img_path,
                                   self.current_ackermann_timestamp,
                                   img_timestamp,
                                   self.current_angle_with_noise])

        cv2.imwrite(img_path, img_to_save)

        self.frame_seq_number += 1

        if self.frame_seq_number % 50 == 0:
            print 'frame_seq_number = ' + str(self.frame_seq_number) \
                  + ', speed = ' + str(self.current_speed) + ', angle = ' \
                  + str(self.current_angle) + ', high_level_command = ' \
                  + str(self.high_level_command) + '.'

    def set_or_reset_hlc_timeout(self, timeout, fn):
        if self.hlc_timer:
            self.hlc_timer.cancel()

        self.hlc_timer = Timer(timeout, fn)
        self.hlc_timer.start()

    def remove_joy_throttle(self):
        self.is_joy_callback_throttled = False

    def _joy_callback(self, joy_message):

        speed_axis = joy_message.axes[SPEED_AXIS]
        angle_axis = joy_message.axes[SERVO_AXIS]
        record_button = joy_message.buttons[RECORD_BUTTON]
        left_hlc_button = joy_message.buttons[LEFT_HLC_BUTTON]
        right_hlc_button = joy_message.buttons[RIGHT_HLC_BUTTON]
        toggle_noise_button = joy_message.buttons[TOGGLE_NOISE_BUTTON]

        has_pressed_relevant_button = bool(record_button) \
                                      or bool(left_hlc_button) or bool(right_hlc_button) or bool(toggle_noise_button)

        # Save speed ang angle values for thread
        self.speedAxisValue = speed_axis
        self.angleAxisValue = angle_axis

        if not has_pressed_relevant_button or self.is_joy_callback_throttled:
            return

        if bool(toggle_noise_button):
            self.noise_enabled = not self.noise_enabled

        self.joy_throttle_timer = Timer(JOY_THROTTLE_SECONDS, self.remove_joy_throttle)
        self.joy_throttle_timer.start()
        self.is_joy_callback_throttled = True

        if bool(record_button):
            if self.recording:
                self.stop_collecting()
            else:
                self.start_collecting()

        if bool(left_hlc_button):
            print 'Setting high_level_command = HLC_LEFT.'
            self.high_level_command = HLC_LEFT
            self.set_or_reset_hlc_timeout(HLC_RESET_SECONDS,
                                          self._reset_hlc)
        elif bool(right_hlc_button):

            print 'Setting high_level_command = HLC_RIGHT.'
            self.high_level_command = HLC_RIGHT
            self.set_or_reset_hlc_timeout(HLC_RESET_SECONDS,
                                          self._reset_hlc)

    def _reset_hlc(self):
        print 'Resetting high_level_command'
        self.high_level_command = HLC_FORWARD

    def publishAckermann(self, speed, angle):

        angleScaled = self._map_range(angle, -1.0, 1.0, -1.0, 1.0)
        speedScaled = self._map_range(speed, -1.0, 1.0, -MAX_SPEED, MAX_SPEED)

        self.current_angle = angleScaled
        self.current_speed = speedScaled
        self.current_ackermann_timestamp = rospy.Time.now().to_nsec()

        if self.noise_enabled:
            if self.current_noise_peak == 0:
                rand = (random.random() * 2 - 1)
                pos = 1 if rand > 0 else -1
                self.current_noise_peak = (MAX_NOISE * rand) + MIN_NOISE * pos
                self.noise_add = self.current_noise_peak / 10
                self.current_noise = 0
                # print("Current noise peak: " + str(self.current_noise_peak))

            self.current_noise += self.noise_add
            if abs(self.current_noise - self.current_noise_peak) < abs(self.current_noise_peak) * 0.01:
                self.noise_add *= -1

            if 0.005 > self.current_noise > -0.005:
                self.current_noise_peak = 0
                self.nosie_add = 0

        # print("Current noise: " + str(self.current_noise))

        self.current_angle_with_noise = angle + self.current_noise

        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.drive.steering_angle = self.current_angle_with_noise
        msg.drive.speed = speedScaled
        msg.drive.acceleration = max_accel_x
        msg.drive.jerk = max_jerk_x

        self.ackermannPub.publish(msg)

    def _thdLoop(self):
        while True:
            self.publishAckermann(self.speedAxisValue, self.angleAxisValue)
            sleep(0.10)

    def _map_range(
            self,
            x,
            in_min,
            in_max,
            out_min,
            out_max,
    ):
        """
        Remaps the values to a new range
        """

        if x is 0:
            return 0
        out = (x - in_min) * (out_max - out_min) / (in_max - in_min) \
              + out_min
        if out > out_max:
            return out_max
        elif out < out_min:
            return out_min
        return out


if __name__ == '__main__':
    rospy.init_node("steer_and_collect_training_data")
    collector = DataCollector()

    print 'Press Start to begin recording...'
    rospy.spin()
