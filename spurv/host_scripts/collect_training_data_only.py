#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import cv2
import errno
import os
from datetime import datetime
from threading import Timer

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from cv_bridge import CvBridge
from expiringdict import ExpiringDict
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Joy

cache = ExpiringDict(max_len=1000, max_age_seconds=2)

RECORD_BUTTON = 7  # Start button
JOY_THROTTLE_SECONDS = 1
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
    csv_field_names = ['speed', 'angle',
                       'image_path', 'ackerman_timestamp']
    folder_name = None
    joystick_subscriber = None
    angle_publisher = None
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
                                   img_path,
                                   self.current_ackerman_timestamp])

        cv2.imwrite(img_path, img_to_save)

        self.frame_seq_number += 1

        if self.frame_seq_number % 50 == 0:
            print 'frame_seq_number = ' + str(self.frame_seq_number) \
                  + ', speed = ' + str(self.current_speed) + ', angle = ' \
                  + str(self.current_angle) + '.'

    def remove_joy_throttle(self):
        self.is_joy_callback_throttled = False

    def _joy_callback(self, joy_message):

        record_button = joy_message.buttons[RECORD_BUTTON]

        has_pressed_relevant_button = bool(record_button)

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


if __name__ == '__main__':
    collector = DataCollector()

    print 'Press Start to begin recording...'
    rospy.spin()
