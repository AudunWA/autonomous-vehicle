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



HLC_RESET_SECONDS = 10
JOY_THROTTLE_SECONDS = 1
NOISE_DURATION_SECONDS = 1

HLC_LEFT = 0
HLC_FORWARD = 1
HLC_RIGHT = 2

IMAGE_SCALE = 0.3


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except OSError, exception:
        if exception.errno != errno.EEXIST:
            raise


class DataCollector:

    STORAGE_DIR = os.path.expanduser('~') + '/data_dump/'

    bridge = CvBridge()
    data_writer = None
    csv_file = None
    csv_field_names = ['speed', 'angle', 'high_level_command',
                       'image_path', 'ackerman_timestamp']
    folder_name = None
    joystick_subscriber = None
    angle_publisher = None
    hlc_timer = None
    joy_throttle_timer = None
    is_joy_callback_throttled = False

    is_adding_noise = False
    current_noise_angle = 0

    def __init__(self):
        print 'Initializing datacollector.'
        self.current_speed = 0
        self.current_angle = 0
        self.current_ackerman_timestamp = 0
        self.recording = False

        rospy.init_node('collect_training_data', anonymous=False)

        self.setup_collection_folders()

        self.csv_file = open(self.folder_name + 'data.csv', 'w+')
        self.data_writer = csv.writer(self.csv_file)

        self.data_writer.writerow(self.csv_field_names)

        self.high_level_command = HLC_FORWARD
        self.frame_seq_number = 0
        self.bridge = CvBridge()

        self.joystick_subscriber = rospy.Subscriber('/joy', Joy,
                self._joy_callback)

    def setup_collection_folders(self):
        self.folder_name = self.STORAGE_DIR \
            + datetime.now().replace(microsecond=0).isoformat() + '/'

        safe_makedirs(self.folder_name)
        safe_makedirs(self.folder_name + 'images/')

    def start_collecting(self):

        print 'Starting data collection. Listening for commands and camera.'

        self.recording = True

        rospy.Subscriber('/ackermann_cmd', AckermannDriveStamped,
                         self._steering_callback)
        rospy.Subscriber('/fwd_camera/image_raw/compressed',
                         CompressedImage, self._image_callback)
        self.ackermann_publisher = rospy.Publisher('/ackermann_cmd',
                AckermannDriveStamped, queue_size=1)

    def stop_collecting(self):
        print 'Stopping data collection. Press Start to continue collecting.'
        self.recording = False

    def get_image_path(self):
        return self.folder_name + 'images/' \
            + str(self.frame_seq_number) + '.jpg'

    def _steering_callback(self, data_loc):
        self.current_speed = data_loc.drive.speed
        self.current_angle = data_loc.drive.steering_angle

    def _image_callback(self, data_loc):

        if not self.recording:
            return

        img_to_save = self.bridge.compressed_imgmsg_to_cv2(data_loc,
                'bgr8')
        img_to_save = cv2.resize(img_to_save, (0, 0), fx=IMAGE_SCALE,
                                 fy=IMAGE_SCALE)

        img_path = self.get_image_path()

        self.data_writer.writerow([self.current_speed,
                                  self.current_angle,
                                  self.high_level_command, img_path,
                                  self.current_ackerman_timestamp])

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

        record_button = joy_message.buttons[RECORD_BUTTON]

        left_hlc_button = joy_message.buttons[LEFT_HLC_BUTTON]
        right_hlc_button = joy_message.buttons[RIGHT_HLC_BUTTON]

        has_pressed_relevant_button = bool(record_button) \
            or bool(left_hlc_button) or bool(right_hlc_button)

        if not has_pressed_relevant_button or self.is_joy_callback_throttled:
            return
            
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
    collector = DataCollector()

    print 'Press Start to begin recording...'
    rospy.spin()


			