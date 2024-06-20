import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def set_camera_settings(cap):
    # Attempt to disable autofocus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 0 to disable, 1 to enable
    # cap.set(cv2.CAP_PROP_FOCUS, 10)  

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 to disable auto exposure
    # cap.set(cv2.CAP_PROP_EXPOSURE, -4)  

    print("Settings set")


def stream_publisher():
    rospy.init_node('camera_stream_publisher', anonymous=True)
    image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
    bridge = CvBridge()

    cap = cv2.VideoCapture('rtmp://192.168.0.4/live/WSL_Stream')

    if not cap.isOpened():
        rospy.logerr("Failed to open RTMP stream")
        return

    set_camera_settings(cap)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Failed to capture frame")
            break

        image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        image_pub.publish(image_msg)

    cap.release()

if __name__ == '__main__':
    try:
        stream_publisher()
    except rospy.ROSInterruptException:
        pass
