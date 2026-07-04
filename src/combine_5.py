#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import io
import kornia as K
import kornia.feature as KF
from kornia.contrib import ImageStitcher
import torch
import kornia.utils as KU
from kornia.io import ImageLoadType
import kornia.image as KI
from kornia.core import Tensor
from typing import Dict, Optional, Tuple
import torch
from kornia.core import Module, Tensor, concatenate, where, zeros_like
from kornia.feature import LocalFeatureMatcher, LoFTR
from kornia.geometry.homography import find_homography_dlt_iterated
from kornia.geometry.ransac import RANSAC
from kornia.geometry.transform import warp_perspective
import functools
import time

import cProfile
import re

calibration_status = 0
left_features = 0
center_features = 0
right_features = 0
counter = 0

homo = [0,0,0]

lir_aspect = 1.0



class ImageStitcher_acg(Module):

  def __init__(self, matcher: Module, estimator: str = "ransac", blending_method: str = "naive") -> None:
    global calibration_status
    #print("__init__")
    super().__init__()
    self.matcher = matcher
    self.estimator = estimator
    self.blending_method = blending_method
    self.ransac = RANSAC("homography")

    '''if estimator not in ["ransac", "vanilla"]:
      raise NotImplementedError(f"Unsupported estimator {estimator}. Use `ransac` or `vanilla` instead.")
    if estimator == "ransac":
      self.ransac = RANSAC("homography")'''
    #print("calibration_status= ", calibration_status )

  def _estimate_homography(self, keypoints1: Tensor, keypoints2: Tensor) -> Tensor:
    """Estimate homography by the matched keypoints.

    Args:
        keypoints1: matched keypoint set from an image, shaped as :math:`(N, 2)`.
        keypoints2: matched keypoint set from the other image, shaped as :math:`(N, 2)`.

    """
    #print("_estimate_homography")
    if self.estimator == "vanilla":
      homo = find_homography_dlt_iterated(
        keypoints2[None], keypoints1[None], torch.ones_like(keypoints1[None, :, 0])
      )
    elif self.estimator == "ransac":
      homo, _ = self.ransac(keypoints2, keypoints1)
      homo = homo[None]
    else:
      raise NotImplementedError(f"Unsupported estimator {self.estimator}. Use `ransac` or `vanilla` instead.")
    return homo

  def estimate_transform(self, *args: Tensor, **kwargs: Tensor) -> Tensor:
    """Compute the corresponding homography."""
    #print("estimate_transform")
    kp1, kp2, idx = kwargs["keypoints0"], kwargs["keypoints1"], kwargs["batch_indexes"]
    homos = [self._estimate_homography(kp1[idx == i], kp2[idx == i]) for i in range(len(idx.unique()))]

    if len(homos) == 0:
      raise RuntimeError("Compute homography failed. No matched keypoints found.")

    #print(concatenate(homos))
    return concatenate(homos)

  def blend_image(self, src_img: Tensor, dst_img: Tensor, mask: Tensor) -> Tensor:
    """Blend two images together."""
    #print("blend_image")
    out: Tensor
    if self.blending_method == "naive":
      out = where(mask == 1, src_img, dst_img)
    else:
      raise NotImplementedError(f"Unsupported blending method {self.blending_method}. Use `naive`.")
    return out

  def preprocess(self, image_1: Tensor, image_2: Tensor) -> Dict[str, Tensor]:
    """Preprocess input to the required format."""
    # TODO: probably perform histogram matching here.
    print("preprocess")
    print("image_0= ", image_1.shape)
    if isinstance(self.matcher, (LoFTR, LocalFeatureMatcher)):
      input_dict = {  # LofTR works on grayscale images only
        "image0": image_1,
        "image1": image_2,
      }
    else:
      raise NotImplementedError(f"The preprocessor for {self.matcher} has not been implemented.")
    print(input_dict["image0"].shape)
    #print("image_0= ", input_dict[0].shape)
    return input_dict

  def postprocess(self, image: Tensor, mask: Tensor) -> Tensor:
    # NOTE: assumes no batch mode. This method keeps all valid regions after stitching.
    #print("postprocess")
    mask_ = mask.sum((0, 1))
    index = int(mask_.bool().any(0).long().argmin().item())
    if index == 0:  # If no redundant space
      return image
    return image[..., :index]

  def on_matcher(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
    #print("on_matcher")
    return self.matcher(data)

  def stitch_pair(
          self,
          images_left: Tensor,
          images_right: Tensor,
          mask_left: Optional[Tensor] = None,
          mask_right: Optional[Tensor] = None,
  ) -> Tuple[Tensor, Tensor]:
    # Compute the transformed images
    global calibration_status, homo, counter
    #print("stitch_pair")
    #print("counter = ", counter)
    #print("calibration_status= ", calibration_status)
    #input_dict = self.preprocess(images_left, images_right)
    input_dict = {"image0": images_left, "image1": images_right}
    out_shape = (images_left.shape[-2], images_left.shape[-1] + images_right.shape[-1])
    if calibration_status == 0:
      correspondences = self.on_matcher(input_dict)
      #print("correspondences = ", correspondences["keypoints0"])
      homo[counter] = self.estimate_transform(**correspondences)
      print(homo[counter])
    src_img = warp_perspective(images_right, homo[counter], out_shape, mode="nearest")
    dst_img = concatenate([images_left, zeros_like(images_right)], -1)
    '''print("homo cuda? ", homo[counter].is_cuda)
    print("images_left cuda? ", images_left.is_cuda)
    print("images_right cuda? ", images_right.is_cuda)
    print("dst_img cuda? ", dst_img.is_cuda)
    print("src_img cuda? ", src_img.is_cuda)'''

    #print("homo= ", type(homo))

    # Compute the transformed masks
    if mask_left is None:
      mask_left = torch.ones_like(images_left)
    if mask_right is None:
      mask_right = torch.ones_like(images_right)
    # 'nearest' to ensure no floating points in the mask
    src_mask = warp_perspective(mask_right, homo[counter], out_shape, mode="nearest")
    dst_mask = concatenate([mask_left, zeros_like(mask_right)], -1)
    #print("stitch_pair complete")
    return self.blend_image(src_img, dst_img, src_mask), (dst_mask + src_mask).bool().to(src_mask.dtype)

  def forward(self, *imgs: Tensor) -> Tensor:
    #print("forward")

    global counter, calibration_status
    img_out = imgs[0]
    mask_left = torch.ones_like(img_out)
    for i in range(len(imgs) - 1):
      img_out, mask_left = self.stitch_pair(img_out, imgs[i + 1], mask_left)
      counter += 1
    #print("for-loop complete")
    calibration_status = 1
    counter = 0

    return self.postprocess(img_out, mask_left)

matcher = KF.LoFTR('outdoor').cuda()  # pretrained='outdoor')
IS = ImageStitcher_acg(matcher, estimator='ransac', blending_method='naive').cuda()

def read_cameras():
  image_pub = rospy.Publisher("combined_image",Image, queue_size = 10)
  
  image_left_sub = message_filters.Subscriber("/camera1/flir_boson_left/image_raw",Image)
  image_center_sub = message_filters.Subscriber("/camera2/flir_boson_center/image_raw",Image)
  image_right_sub = message_filters.Subscriber("/camera3/flir_boson_right/image_raw",Image)

  ts = message_filters.ApproximateTimeSynchronizer([image_left_sub, image_center_sub, image_right_sub], queue_size=10, slop=0.02)
  ts.registerCallback(image_callback)
  rospy.spin()

def image_callback(imageL, imageC, imageR):
  br = CvBridge()
  #rospy.loginfo("receiving frames")
  imageLeft = br.imgmsg_to_cv2(imageL, desired_encoding='passthrough')
  imageCenter = br.imgmsg_to_cv2(imageC,  desired_encoding='passthrough')
  imageRight = br.imgmsg_to_cv2(imageR,  desired_encoding='passthrough')

  tensorleft = KU.image_to_tensor(imageLeft/256, keepdim=False).cuda().float()
  tensorcenter = KU.image_to_tensor(imageCenter/256, keepdim=False).cuda().float()
  tensorright = KU.image_to_tensor(imageRight/256, keepdim=False).cuda().float()

  imgs = [tensorleft, tensorcenter, tensorright]

  #print(imgs[0].shape)

  #start_time = time.time()

  with torch.inference_mode():
      out = IS(*imgs)

  #stop_time = time.time()
  #print("********** duration= ", stop_time - start_time)

  #out = andy_stitching_func(*imgs)

  out1 = out*255
  out2 = K.tensor_to_image(out1)
  out3 = out2.astype(np.uint8)

  #out3 = br.cv2_to_imgmsg(out3, 'mono8')
  #out3 = br.imgmsg_to_cv2(out3, desired_encoding='passthrough')

  image_pub = rospy.Publisher("combined_image",Image, queue_size = 2)

  try:
    image_pub.publish(br.cv2_to_imgmsg(out3, 'mono8'))
  except CvBridgeError as e:
    print(e)

def main(args):
  rospy.init_node('combined_image', anonymous=True)

  try:
    read_cameras()
    #rospy.spin()
  except KeyboardInterrupt:
    print ("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)



'''
def load_images(fnames):
   #return [K.io.load_image(fn, ImageLoadType.RGB32, device="cuda")[None, ...] for fn in fnames]
   return [K.io.load_image(fn, ImageLoadType.RGB32, device="cuda")[None, ...] for fn in fnames]

def dnu_estimate_transform(*args: Tensor, **kwargs: Tensor) -> Tensor:
  """Compute the corresponding homography."""

  kp1, kp2, idx = kwargs["keypoints0"], kwargs["keypoints1"], kwargs["batch_indexes"]
  homos = [self._estimate_homography(kp1[idx == i], kp2[idx == i]) for i in range(len(idx.unique()))]

  if len(homos) == 0:
    raise RuntimeError("Compute homography failed. No matched keypoints found.")
  return concatenate(homos)

def estimate_transform(self, **kwargs: Tensor) -> Tensor:
  """Compute the corresponding homography."""

  kp1, kp2, idx = kwargs["keypoints0"], kwargs["keypoints1"], kwargs["batch_indexes"]
  homos = [self._estimate_homography(kp1[idx == i], kp2[idx == i]) for i in range(len(idx.unique()))]

  if len(homos) == 0:
    raise RuntimeError("Compute homography failed. No matched keypoints found.")
  return concatenate(homos)

def stitch_pair(
        self,
        images_left: Tensor,
        images_right: Tensor,
        mask_left: Optional[Tensor] = None,
        mask_right: Optional[Tensor] = None,
        ) -> Tuple[Tensor, Tensor]:

  input_dict = {"image0": images_left, "image1": images_right}
  out_shape = (images_left.shape[-2], images_left.shape[-1] + images_right.shape[-1])
  if calibration_status == 0:
    #correspondences = on_matcher(input_dict)
    print("input_dict = ", input_dict)
    loftr = KF.LoFTR('outdoor').cuda()
    #print("loftr = ", loftr)
    correspondences = loftr(input_dict)
    print("correspondences = ", correspondences["keypoints0"])
    homo[self.counter] = estimate_transform(**correspondences)
    #kp1, kp2, idx = correspondences["keypoints0"], correspondences["keypoints1"], correspondences["batch_indexes"]
    #self.homo = [self._estimate_homography(kp1[idx == i], kp2[idx == i]) for i in range(len(idx.unique()))]
  print("counter = ", counter)
  print("homo = ", self.homo[0])
  src_img = warp_perspective(images_right, self.homo[counter], out_shape, mode="nearest")
  dst_img = concatenate([images_left, zeros_like(images_right)], -1)

  # Compute the transformed masks
  if mask_left is None:
    mask_left = torch.ones_like(images_left)
  if mask_right is None:
    mask_right = torch.ones_like(images_right)
  # 'nearest' to ensure no floating points in the mask
  src_mask = warp_perspective(mask_right, self.homo[counter], out_shape, mode="nearest")
  dst_mask = concatenate([mask_left, zeros_like(mask_right)], -1)
  #print("stitch_pair complete")
  return self.blend_image(src_img, dst_img, src_mask), (dst_mask + src_mask).bool().to(src_mask.dtype)

def andy_stitching_func(self, *imgs: Tensor, estimator: str = "ransac", blending_method: str = "naive") -> Tensor:
  global homo, calibration_status, counter
  self.matcher = KF.LoFTR('outdoor').cuda()
  self.estimator = estimator
  self.blending_method = blending_method
  self.ransac = RANSAC("homography")
  self.homo = homo
  counter = 0

  img_out = imgs[0]
  mask_left = torch.ones_like(img_out)
  for i in range(len(imgs) - 1):
    img_out, mask_left = stitch_pair(img_out, imgs[i + 1], mask_left)
    counter += 1

  # print("for-loop complete")
  if calibration_status == 0:
    self.homo = homo
  calibration_status = 1
  return self.postprocess(img_out, mask_left)
'''