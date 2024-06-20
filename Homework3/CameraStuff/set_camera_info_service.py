#!/usr/bin/env python3

import rospy
from sensor_msgs.srv import SetCameraInfo, SetCameraInfoResponse

def handle_set_camera_info(req):
    rospy.loginfo("SetCameraInfo request received")
    response = SetCameraInfoResponse()
    response.success = True
    response.status_message = "Camera info set successfully"
    return response

def set_camera_info_server():
    rospy.init_node('set_camera_info_server')
    service = rospy.Service('/camera/set_camera_info', SetCameraInfo, handle_set_camera_info)
    rospy.loginfo("SetCameraInfo service ready")
    rospy.spin()

if __name__ == "__main__":
    set_camera_info_server()
