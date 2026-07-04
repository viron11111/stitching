#!/usr/bin/env python3

import roslib

import sys
import rospy
import cv2
import numpy as np
import math
#import cv2.cv as cv
import colorsys
import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import message_filters

from matplotlib import pyplot as plt

from stitching import Stitcher
from stitching.images import Images
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.timelapser import Timelapser
from stitching.cropper import Cropper
from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender
from stitching import AffineStitcher


seam_finder = SeamFinder()
cropper = Cropper()
subsetter = Subsetter()
camera_estimator = CameraEstimator()
camera_adjuster = CameraAdjuster()
wave_corrector = WaveCorrector()
stitcher = Stitcher()
warper = Warper()


calibration_status = 0
left_features = 0
center_features = 0
right_features = 0


def set_cal_to_zero():
  global calibration_status
  calibration_status = 0

def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def read_cameras():
  image_pub = rospy.Publisher("combined_image",Image, queue_size = 1)
  

  
  #rospy.init_node('combined_image', anonymous=True)
  
  #image_left_sub = message_filters.Subscriber("/camera1/usb_cam_left/image_raw",Image)
  #image_center_sub = message_filters.Subscriber("/camera2/usb_cam_center/image_raw",Image)
  #image_right_sub = message_filters.Subscriber("/camera3/usb_cam_right/image_raw",Image)

  image_left_sub = message_filters.Subscriber("/camera1/flir_boson_left/image_raw",Image)
  image_center_sub = message_filters.Subscriber("/camera2/flir_boson_center/image_raw",Image)
  image_right_sub = message_filters.Subscriber("/camera3/flir_boson_right/image_raw",Image)

  ts = message_filters.ApproximateTimeSynchronizer([image_left_sub, image_center_sub, image_right_sub], queue_size=10, slop=0.5)
  ts.registerCallback(image_callback)
  rospy.spin()

def image_callback(imageL, imageC, imageR):
  global calibration_status, left_features, center_features, right_features, matches, features,\
      cameras, warped_final_imgs, final_corners, final_sizes, warped_low_imgs, warped_low_masks,\
      low_corners, low_sizes, cropped_low_imgs, cropped_low_masks, cropped_final_masks,\
      cropped_final_imgs, seam_masks

  br = CvBridge()
  rospy.loginfo("receiving frames")
  imageLeft = br.imgmsg_to_cv2(imageL)
  imageCenter = br.imgmsg_to_cv2(imageC)
  imageRight = br.imgmsg_to_cv2(imageR)

  #imageLeft = cv2.cvtColor(br.imgmsg_to_cv2(imageL), cv2.COLOR_BGR2GRAY)
  #imageCenter = cv2.cvtColor(br.imgmsg_to_cv2(imageC), cv2.COLOR_BGR2GRAY)
  #imageRight = cv2.cvtColor(br.imgmsg_to_cv2(imageR), cv2.COLOR_BGR2GRAY)

  ### for FLIR images which are mono8 ###############################
  imageLeft = cv2.cvtColor(imageLeft, cv2.COLOR_GRAY2RGB)
  imageCenter = cv2.cvtColor(imageCenter, cv2.COLOR_GRAY2RGB)
  imageRight = cv2.cvtColor(imageRight, cv2.COLOR_GRAY2RGB)
  ###################################################################



  settings = {"detector": "orb", "confidence_threshold": 1.0, "nfeatures": 10000, "try_use_gpu": True, "crop": True\
  }#, "compensator": "no", "blender_type": "no", "finder": "no"}
  stitcher = Stitcher(**settings)

  panorama = stitcher.stitch([imageLeft, imageCenter, imageRight], )  # the user is warned that only a subset of input images is stitched
  rospy.loginfo("line 113")


  image_pub = rospy.Publisher("combined_image",Image, queue_size = 20)

  try:
    #image_pub.publish(br.cv2_to_imgmsg(keypoints_all, "rgb8")) 
    image_pub.publish(br.cv2_to_imgmsg(panorama, "rgb8")) 
    #rospy.loginfo("published image")
  except CvBridgeError as e:
    print(e)

def seam_masks_func():
  global cropped_low_imgs, low_corners, cropped_low_masks, cropped_final_masks, cropped_final_imgs,\
    seam_masks
  seam_finder = SeamFinder()

  seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
  seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_final_masks)]

  seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(cropped_final_imgs, seam_masks)]
  #plot_images(seam_masks_plots, (15,10))

def cropper_func():
  global warped_final_imgs, warped_final_masks, final_corners, final_sizes, low_corners, low_sizes,\
    cropped_low_imgs, cropped_low_masks, cropped_final_masks, cropped_final_imgs

  cropper = Cropper()
  mask = cropper.estimate_panorama_mask(warped_final_imgs, warped_final_masks, final_corners, final_sizes)
  lir = cropper.estimate_largest_interior_rectangle(mask)
  plot = lir.draw_on(mask, size=2)
  #plot_image(plot, (5,5))
  low_corners = cropper.get_zero_center_corners(low_corners)
  rectangles = cropper.get_rectangles(low_corners, low_sizes)

  plot = rectangles[1].draw_on(plot, (0, 255, 0), 2)  # The rectangle of the center img
  #plot_image(plot, (5,5))
  overlap = cropper.get_overlap(rectangles[1], lir)
  plot = overlap.draw_on(plot, (255, 0, 0), 2)
  #plot_image(plot, (5,5))
  intersection = cropper.get_intersection(rectangles[1], overlap)
  plot = intersection.draw_on(warped_low_masks[1], (255, 0, 0), 2)
  #plot_image(plot, (2.5,2.5))


  cropper.prepare(warped_final_imgs, warped_final_masks, final_corners, final_sizes)
  cropped_low_masks = list(cropper.crop_images(warped_final_masks))
  cropped_low_imgs = list(cropper.crop_images(warped_final_imgs))
  low_corners, low_sizes = cropper.crop_rois(final_corners, final_sizes)

  lir_aspect = 1.0  #images.get_ratio(Images.Resolution.FINAL, Images.Resolution.FINAL)  # since lir was obtained on low imgs
  cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
  cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
  final_corners, final_sizes = cropper.crop_rois(final_corners, final_sizes, lir_aspect)

  timelapser = Timelapser('as_is')
  timelapser.initialize(final_corners, final_sizes)

  #for img, corner in zip(cropped_final_imgs, final_corners):
  #    timelapser.process_frame(img, corner)
  #    frame = timelapser.get_frame()
  #    plot_image(frame, (10,10))


def warp_images(cameras, imageLeft, imageCenter, imageRight):
  global warped_final_imgs, final_corners, final_sizes, warped_low_imgs, warped_low_masks,\
   low_corners, low_sizes, warped_final_masks

  warper.set_scale(cameras)
  images = Images.of([imageLeft, imageCenter, imageRight])
  low_imgs = list(images.resize(Images.Resolution.LOW))
  final_imgs = list(images.resize(Images.Resolution.FINAL))
  low_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)

  camera_aspect = 1.0 #images.get_ratio(Images.Resolution.FINAL, Images.Resolution.FINAL)  # since cameras were obtained on medium imgs

  #rospy.loginfo(camera_aspect)

  warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
  warped_low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
  low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

  final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
  camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

  warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
  warped_final_masks = list(warper.create_and_warp_masks(final_sizes, cameras, camera_aspect))
  final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

  #lot_images(warped_low_imgs, (10,10))
  #plot_images(warped_low_masks, (10,10))


def matching_generator(feat, imgL, imgC, imgR):
  global matches

  matcher = FeatureMatcher()
  matches = matcher.match_features(feat)

  '''all_relevant_matches = matcher.draw_matches_matrix([imgL, imgC, imgR], feat, matches, conf_thresh=1, 
                                                     inliers=True, matchColor=(0, 255, 0))

  #len_matches = len(all_relevant_matches.shape)
  #rospy.loginfo(len_matches)

  for idx1, idx2, img in all_relevant_matches:
      print(f"Matches Image {idx1+1} to Image {idx2+1}")
      plot_image(img, (20,10))'''

  return matches


def keypoint_generator(imgL, imgC, imgR):

  global left_features, center_features, right_features

  left_mask = np.zeros(imgL.shape[:2], np.uint8)
  #cv2.rectangle(left_mask, (400, 0), (640, 480), 255, -1)
  cv2.rectangle(left_mask, (0, 0), (640, 480), 255, -1)
  
  finder = FeatureDetector(nfeatures=500000)
  left_features = finder.detect_features(imgL, left_mask)   


  center_mask = np.zeros(imgC.shape[:2], np.uint8)
  
  cv2.rectangle(center_mask, (0, 0), (640, 480), 255, -1)  
  #cv2.rectangle(center_mask, (400, 0), (640, 480), 255, -1)
  #cv2.rectangle(center_mask, (0, 0), (200, 480), 255, -1)
  
  center_features = finder.detect_features(imgC, center_mask)

  right_mask = np.zeros(imgR.shape[:2], np.uint8)
  #cv2.rectangle(right_mask, (0, 0), (200, 480), 255, -1)
  cv2.rectangle(right_mask, (0, 0), (640, 480), 255, -1)
  
  right_features = finder.detect_features(imgR, right_mask)

  features = [left_features, center_features, right_features]
  return features



  #**************************************

  '''

  panorama = stitcher.stitch([imageLeft, imageCenter, imageRight])

  #rospy.loginfo("performed stitching")

  #vid = np.concatenate((imageLeft, imageRight), axis=1)

  image_pub = rospy.Publisher("combined_image",Image, queue_size = 1)

  #rospy.loginfo("prepping image")

  try:
    image_pub.publish(br.cv2_to_imgmsg(panorama, "rgb8")) 
    #rospy.loginfo("published image")
  except CvBridgeError as e:
    print(e)
  '''


def main(args):
  #ci = combine_image_class()

  rospy.init_node('combined_image', anonymous=True)

  set_cal_to_zero()

  try:
    read_cameras()
    #rospy.spin()
  except KeyboardInterrupt:
    print ("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
