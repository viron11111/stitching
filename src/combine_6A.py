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

import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

from stitching.cropper import Cropper

import cProfile
import re

from stitching.cropper import Cropper

cropper = Cropper()

calibration_status = 0
left_features = 0
center_features = 0
right_features = 0
counter = 0

homo = [0,0,0]

lir_aspect = 1.0

cropper = Cropper()

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

    src_img = warp_perspective(images_right, homo[counter], out_shape, mode="nearest")
    #src_img = warp_perspective(images_right, M, out_shape, mode="nearest")
    dst_img = concatenate([images_left, zeros_like(images_right)], -1)

    #image_out = src_img.cpu().numpy()
    #print(image_out[0][0].shape)
    #plt.imshow(image_out[0][0])
    #plt.show()

    # Compute the transformed masks
    if mask_left is None:
      mask_left = torch.ones_like(images_left)
    if mask_right is None:
      mask_right = torch.ones_like(images_right)
    # 'nearest' to ensure no floating points in the mask



    src_mask = warp_perspective(mask_right, homo[counter], out_shape, mode="nearest")
    #src_mask = warp_perspective(mask_right, heart_mask, out_shape, mode="nearest")

    image_out = src_mask.cpu().numpy()
    #print(image_out[0][0].shape)
    #plt.imshow(image_out[0][0])
    #plt.show()

    dst_mask = concatenate([mask_left, zeros_like(mask_right)], -1)



    #plt.imshow(lir)
    #plt.show()

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

    if calibration_status == 1:

      plot_mask = mask_left[0][0].cpu().numpy()
      plot_img = img_out[0][0].cpu().numpy()

      plot_mask = plot_mask.astype(np.uint8)
      plot_img = plot_mask.astype(np.uint8)

      #plt.imshow(plot_mask)
      #plt.show()

      lir = cropper.estimate_largest_interior_rectangle(plot_mask)

      print(lir)

      #low_corners = [(0, 0), (385, 0), (757, 0)]
      #low_sizes = [(635, 475), (639, 475), (637, 475)]

      low_corners = [(0, 0), (757, 0)]
      low_sizes = [(635, 475), (637, 475)]

      low_corners = cropper.get_zero_center_corners(low_corners)
      rectangles = cropper.get_rectangles(low_corners, low_sizes)

      overlap = cropper.get_overlap(rectangles[1], lir)

      #intersection = cropper.get_intersection(rectangles[1], overlap)

      cropper.prepare(plot_img, plot_mask, low_corners, low_sizes)

      cropped_low_masks = list(cropper.crop_images(plot_mask))
      cropped_low_imgs = list(cropper.crop_images(plot_img))
      low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)



      calibration_status = 2


    counter = 0

    return self.postprocess(img_out, mask_left)

matcher = KF.LoFTR('outdoor').cuda()  # pretrained='outdoor')
IS = ImageStitcher_acg(matcher, estimator='ransac', blending_method='naive').cuda()

def image_callback():
  #br = CvBridge()

  fnames = ["imageLeft.jpg", "imageCenter.jpg", "imageRight.jpg"]

  imgs = [K.io.load_image(fn, K.io.ImageLoadType.GRAY8, device='cuda')[None, ...] for fn in fnames]

  #print(imgs[0].shape)
  #print(imgs[1].shape)
  #print(imgs[2].shape)

  imgs = [imgs[0]/255, imgs[1]/255, imgs[2]/255]

  #print(imgs[0].shape)

  with torch.inference_mode():
      out = IS(*imgs)

  out1 = out*255
  out2 = K.tensor_to_image(out1)
  out3 = out2.astype(np.uint8)

  plt.imshow(out3)
  plt.show()

def main(args):
  image_callback()

if __name__ == '__main__':
    main(sys.argv)
'''
[tensor([[[[0.2930, 0.6094, 0.6797,  ..., 0.6250, 0.6641, 0.6914],
          [0.6953, 0.7344, 0.7578,  ..., 0.7695, 0.8008, 0.8242],
          [0.7539, 0.8047, 0.8672,  ..., 0.8008, 0.8008, 0.8320],
          ...,
          [0.7891, 0.8242, 0.8086,  ..., 0.6758, 0.6680, 0.6641],
          [0.7969, 0.8164, 0.8008,  ..., 0.5859, 0.5859, 0.5703],
          [0.8008, 0.7969, 0.8086,  ..., 0.5000, 0.5234, 0.5195]]]],
       device='cuda:0'), tensor([[[[0.6602, 0.7109, 0.6055,  ..., 0.2969, 0.3477, 0.3359],
          [0.6719, 0.6836, 0.5859,  ..., 0.4023, 0.5039, 0.4414],
          [0.6094, 0.6602, 0.5625,  ..., 0.3867, 0.5547, 0.4805],
          ...,
          [0.6875, 0.7344, 0.6328,  ..., 0.3516, 0.4844, 0.4180],
          [0.7031, 0.7383, 0.6055,  ..., 0.3477, 0.4688, 0.4453],
          [0.7109, 0.7109, 0.6211,  ..., 0.3555, 0.4922, 0.4414]]]],
       device='cuda:0'), tensor([[[[0.8164, 0.8242, 0.8672,  ..., 0.0703, 0.1172, 0.1250],
          [0.8008, 0.8086, 0.8398,  ..., 0.0703, 0.1211, 0.1055],
          [0.7266, 0.6992, 0.7539,  ..., 0.1055, 0.1055, 0.1328],
          ...,
          [0.6328, 0.6641, 0.7031,  ..., 0.7578, 0.8008, 0.8672],
          [0.5352, 0.6367, 0.6836,  ..., 0.8164, 0.8555, 0.8633],
          [0.5078, 0.5781, 0.6562,  ..., 0.8438, 0.8633, 0.8516]]]],
       device='cuda:0')]
'''


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
  
  *****************************************************************************
  
      img_out = imgs[0]
    mask_left = torch.ones_like(img_out)
    print(img_out.shape)

    mask_left = img_out.cpu().numpy()
    print(mask_left.shape)
    print("zero")
    print(mask_left[0,2].shape)
    print(mask_left[0,2])

    image = mask_left[0,2]
    print(image.shape)
    print(image)

    plt.imshow(image/255)

    #image_ts = np.reshape(image_ts, (1,2))

    #plt.imshow(image_ts[1,3])
    plt.show()
  
      image_out = img_out.cpu().numpy()
      print(image_out[0][0].shape)
      plt.imshow(image_out[0][0])
      plt.show()
  
  
    print(homo[counter])
    M = torch.eye(3)[None]
    M = M.to(device='cuda')
    print(M)
    
    
        heart = imread(r'heart.jpg', cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(heart, thresh=180, maxval=255, type=cv2.THRESH_BINARY)

    mask_match = image_out[0][0]
    print(mask_match.shape)

    src_x, src_y = mask_match.shape
    heart_x, heart_y = mask.shape

    x_heart = min(src_x, heart_x)
    x_half_heart = mask.shape[0] // 2

    heart_mask = mask[x_half_heart - x_heart // 2: x_half_heart + x_heart // 2 + 1, :src_y]
    #plt.imshow(heart_mask, cmap='Greys_r')
    #plt.show()

    heart_mask = torch.from_numpy(heart_mask)
    
        #**************************************************************************

    heart = imread(r'heart.jpg', cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(heart, thresh=180, maxval=255, type=cv2.THRESH_BINARY)

    mask_match = image_out[0][0]
    print(mask_match.shape)

    src_x, src_y = mask_match.shape
    heart_x, heart_y = mask.shape

    print(mask.shape)

    x_heart = min(src_x, heart_x)
    print("x_heart= ", x_heart)
    x_half_heart = mask.shape[0] // 2

    print("x_half_heart= ", x_half_heart)

    heart_mask = mask[x_half_heart - x_heart // 2: x_half_heart + x_heart // 2 + 1, :src_y]

    print("heart_maks.shape= ", heart_mask.shape)

    #plt.imshow(heart_mask, cmap='Greys_r')
    #plt.show()

    #heart_mask = torch.from_numpy(heart_mask)

    mask_match_width_half = mask_match.shape[1] // 2
    print(mask_match_width_half)
    mask_match_to_mask = mask_match[:, mask_match_width_half - x_half_heart:mask_match_width_half + x_half_heart]
    #mask_match_to_mask = mask_match_to_mask.astype(np.uint8)

    print(type(mask_match_to_mask))
    print(type(heart_mask))

    print(mask_match_to_mask.dtype)
    print(heart_mask.dtype)

    mask_match_to_mask = mask_match_to_mask.astype(np.uint8)

    print(mask_match_to_mask.dtype)
    print(heart_mask.dtype)

    #plt.imshow(mask_match_to_mask)
    #plt.show()

    masked = cv2.bitwise_and(mask_match_to_mask, mask_match_to_mask, mask=heart_mask)
    plt.imshow(masked)
    plt.show()


    #******************************************************************************************************
    
    
'''