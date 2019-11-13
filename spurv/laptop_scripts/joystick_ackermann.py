#!/usr/bin/env python
"""
Control the spurv with a joystick
"""

import rospy
import cv2
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64
from spurv_msgs.srv import ToggleGpio
from imageconverter import RosToOpenCvConverter
from threading import Thread
from time import sleep
from ackermann_msgs.msg import AckermannDriveStamped

SPEED_AXIS=3
SERVO_AXIS=0
LIGHTS_BUTTON=2

MAX_SPEED=3

frame_id = rospy.get_param("~frame_id", "odom")
max_accel_x = rospy.get_param("~acc_lim_x" , 1.0)
max_jerk_x = rospy.get_param("~jerk_lim_x" , 0.0)

class SpurvJoystickDriver(object):

    speedAxisValue = 0.0
    angleAxisValue = 0.0
    
    def __init__(self):
        self._initPubSub()
        rospy.loginfo("Spurv Joystick Driver initialized")
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
        self.joystickSubscriber = rospy.Subscriber("/joy",
                                                  Joy,
                                                  self._onJoyCallback)

        self.ackermannPub = rospy.Publisher("/ackermann_cmd", AckermannDriveStamped, queue_size=1)
        
    def _onJoyCallback(self, joyMessage):
        """
        Handle joystick messages
        """

        speedAxis = joyMessage.axes[SPEED_AXIS]
        angleAxis = joyMessage.axes[SERVO_AXIS]


        # Save speed ang angle values for thread
        self.speedAxisValue = speedAxis
        self.angleAxisValue = angleAxis


    def publishAckermann(self, speed, angle):

        angleScaled = self._map_range(angle, -1.0, 1.0, -1.0, 1.0)
        speedScaled = self._map_range(speed, -1.0, 1.0, -MAX_SPEED, MAX_SPEED)
        
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.drive.steering_angle = angleScaled
        msg.drive.speed = speedScaled
        msg.drive.acceleration = max_accel_x
        msg.drive.jerk = max_jerk_x

        self.ackermannPub.publish(msg)


    def _thdLoop(self):
        while(True):

	    self.publishAckermann(self.speedAxisValue, self.angleAxisValue)
            sleep(0.10)

    def _map_range(self, x, in_min, in_max, out_min, out_max):
        """
        Remaps the values to a new range
        """
        if x is 0:
            return 0
        out = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        if out > out_max:
            return out_max
        elif out < out_min:
            return out_min
        return out

def onFwdCameraImage(cvImage):
    """
    Camera Callback with opencv image
    """
    cv2.imshow("Camera", cvImage)
    cv2.waitKey(1)
    
if __name__ == '__main__':
    rospy.init_node("spurv_joy_driver")
    driver = SpurvJoystickDriver()
    camera = RosToOpenCvConverter("/fwd_camera/image_raw", onFwdCameraImage)

    try:
        rospy.spin()
    except KeyboardInterrupt as interrupt:
        pass
    

