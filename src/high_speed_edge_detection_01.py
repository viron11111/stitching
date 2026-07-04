#!/usr/bin/env python3
import kornia.filters
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import kornia as K
import kornia.utils as KU

#setup publisher and subscriber
def read_cameras():
  image_pub = rospy.Publisher("edge_detect",Image, queue_size = 10)

  #flir camera nodes, 30 fps
  image_left_sub = message_filters.Subscriber("/camera1/flir_boson_left/image_raw",Image)
  image_center_sub = message_filters.Subscriber("/camera2/flir_boson_center/image_raw",Image)
  image_right_sub = message_filters.Subscriber("/camera3/flir_boson_right/image_raw",Image)

  #Applying synchronizer to ensure synchronized frames are grabbed.
  ts = message_filters.ApproximateTimeSynchronizer([image_left_sub, image_center_sub, image_right_sub], queue_size=10, slop=0.02)
  ts.registerCallback(image_callback)
  rospy.spin()

#main function
def image_callback(imageL, imageC, imageR):
  #Ceeded for OpenCV
  br = CvBridge()

  #Convert ROS image message to OpenCV, retain file type
  imageLeft = br.imgmsg_to_cv2(imageL, desired_encoding='passthrough')
  imageCenter = br.imgmsg_to_cv2(imageC,  desired_encoding='passthrough')
  imageRight = br.imgmsg_to_cv2(imageR,  desired_encoding='passthrough')

  #Convert CV2 images to tensor and store in GPU memory
  tensorleft = KU.image_to_tensor(imageLeft/256, keepdim=False).cuda().float()
  tensorcenter = KU.image_to_tensor(imageCenter/256, keepdim=False).cuda().float()
  tensorright = KU.image_to_tensor(imageRight/256, keepdim=False).cuda().float()

  #Place in list
  imgs = [tensorleft, tensorcenter, tensorright]

  # Define Kornia Canny filter
  #https://kornia.readthedocs.io/en/latest/filters.html
  canny = kornia.filters.Canny(kernel_size=[5,5], sigma=(2.5,2.5), hysteresis=False, eps=1e-6).cuda()

  #Apply Kornia filter to camera images
  values = [canny(image) for image in imgs]

  #Values consists of float values from 0.0-1.0.  Need to convert to 8-bit int
  tensor_byte = [values[x][1]*255 for x in range(len(imgs))]
  #Convert tensor to image
  back_to_image = [K.tensor_to_image(tensor_byte[x]) for x in range(len(imgs))]
  #Convert images to message type acceptable by ROS
  edge_images = [back_to_image[x].astype(np.uint8) for x in range(len(imgs))]

  #Concatenate three feeds horizontally into one image
  combined_edge_images = cv2.hconcat(edge_images)

  #Define message to be published
  image_pub = rospy.Publisher("edge_detect",Image, queue_size = 2)

  try:
    #Publish image
    image_pub.publish(br.cv2_to_imgmsg(combined_edge_images, 'mono8'))
  except CvBridgeError as e:
    print(e)

def main(args):
  rospy.init_node('edge_detect', anonymous=True)

  try:
    read_cameras()
    #rospy.spin()
  except KeyboardInterrupt:
    print ("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)