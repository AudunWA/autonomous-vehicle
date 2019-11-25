import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class RosToOpenCvConverter(object):
    """
    Class for converting a image sent over image transport
    to opencv images
    """
    
    def __init__(self, imageTopic, imageCallback):
        self.imageTopic = imageTopic
        self.imageCallback = imageCallback
        self.cvBridge = CvBridge()
        self.subscribeToTopic(imageTopic)
        
    def subscribeToTopic(self, topic):
        """
        Subscribes to the specified image
        """
        self.imageSubscriber = rospy.Subscriber(topic,
                                                Image,
                                                self._imageReceived)

    def _imageReceived(self, image):
        """
        Callback for ros subscriber. Whenever a image is received this
        function is called with a reference to the image.
        """
        cvImage = self._convertToOpenCv(image)
        self.imageCallback(cvImage)

    def _convertToOpenCv(self, data):
        """
        Converts the image data to opencv image format
        """
        try:
	    # OpenCV gets the format wrong, therefore "bgr8" is specified here
            # Else, use data.encoding
            #cvImage = self.cvBridge.imgmsg_to_cv2(data, data.encoding)
	    cvImage = self.cvBridge.imgmsg_to_cv2(data, "bgr8")
            return cvImage
        except CvBridgeError as e:
            print(e)
