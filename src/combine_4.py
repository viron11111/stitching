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
    print("__init__")
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
    print("_estimate_homography")
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
    print("estimate_transform")
    kp1, kp2, idx = kwargs["keypoints0"], kwargs["keypoints1"], kwargs["batch_indexes"]
    homos = [self._estimate_homography(kp1[idx == i], kp2[idx == i]) for i in range(len(idx.unique()))]

    if len(homos) == 0:
      raise RuntimeError("Compute homography failed. No matched keypoints found.")

    #print(concatenate(homos))
    return concatenate(homos)

  def blend_image(self, src_img: Tensor, dst_img: Tensor, mask: Tensor) -> Tensor:
    """Blend two images together."""
    print("blend_image")
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
    print("postprocess")
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
    print("stitch_pair")
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
    print("stitch_pair complete")
    return self.blend_image(src_img, dst_img, src_mask), (dst_mask + src_mask).bool().to(src_mask.dtype)

  def forward(self, *imgs: Tensor) -> Tensor:
    print("forward")
    global counter, calibration_status
    img_out = imgs[0]
    mask_left = torch.ones_like(img_out)
    for i in range(len(imgs) - 1):
      img_out, mask_left = self.stitch_pair(img_out, imgs[i + 1], mask_left)
      counter += 1

    print("for-loop complete")
    calibration_status = 1
    counter = 0
    return self.postprocess(img_out, mask_left)

def load_images(fnames):
   #return [K.io.load_image(fn, ImageLoadType.RGB32, device="cuda")[None, ...] for fn in fnames]
   return [K.io.load_image(fn, ImageLoadType.RGB32, device="cuda")[None, ...] for fn in fnames]

def read_cameras():
  image_pub = rospy.Publisher("combined_image",Image, queue_size = 5)
  
  image_left_sub = message_filters.Subscriber("/camera1/flir_boson_left/image_raw",Image)
  image_center_sub = message_filters.Subscriber("/camera2/flir_boson_center/image_raw",Image)
  image_right_sub = message_filters.Subscriber("/camera3/flir_boson_right/image_raw",Image)

  #ts = message_filters.ApproximateTimeSynchronizer([image_left_sub, image_center_sub, image_right_sub], queue_size=10, slop=0.016)
  ts = message_filters.ApproximateTimeSynchronizer([image_left_sub, image_center_sub, image_right_sub], queue_size=10, slop=0.04)
  ts.registerCallback(image_callback)
  rospy.spin()

def image_callback(imageL, imageC, imageR):

  global calibration_status

  br = CvBridge()
  rospy.loginfo("receiving frames")
  imageLeft = br.imgmsg_to_cv2(imageL, desired_encoding='passthrough')
  imageCenter = br.imgmsg_to_cv2(imageC,  desired_encoding='passthrough')
  imageRight = br.imgmsg_to_cv2(imageR,  desired_encoding='passthrough')

  #image_left = cv2.cvtColor(imageLeft, cv2.COLOR_GRAY2RGB)/256
  #image_center = cv2.cvtColor(imageCenter, cv2.COLOR_GRAY2RGB)/256
  #image_right = cv2.cvtColor(imageRight, cv2.COLOR_GRAY2RGB)/256

  #tensorleft = KU.image_to_tensor(image_left, keepdim=False).cuda().float()
  #tensorcenter = KU.image_to_tensor(image_center, keepdim=False).cuda().float()
  #tensorright = KU.image_to_tensor(image_right, keepdim=False).cuda().float()

  tensorleft = KU.image_to_tensor(imageLeft/256, keepdim=False).cuda().float()
  tensorcenter = KU.image_to_tensor(imageCenter/256, keepdim=False).cuda().float()
  tensorright = KU.image_to_tensor(imageRight/256, keepdim=False).cuda().float()

  #print("image_type: \n", type(tensorleft))
  #print("variable_type: \n", tensorleft.dtype)
  #print("shape: \n", tensorleft.shape)
  #print(tensorleft)

  '''cv2.imwrite('imageLeft.jpg', imageLeft)
  cv2.imwrite('imageCenter.jpg', imageCenter)
  cv2.imwrite('imageRight.jpg', imageRight)'''

  ### for FLIR images which are mono8 ###############################
  #imageLeft = KU.image_to_tensor(cv2.cvtColor(imageLeft, cv2.COLOR_GRAY2RGB), keepdim = False).cuda().float()
  #imageCenter = KU.image_to_tensor(cv2.cvtColor(imageCenter, cv2.COLOR_GRAY2RGB), keepdim = False).cuda().float()
  #imageRight = KU.image_to_tensor(cv2.cvtColor(imageRight, cv2.COLOR_GRAY2RGB), keepdim = False).cuda().float()
  ###################################################################

  #imageLeft_rgb = cv2.cvtColor(imageLeft, cv2.COLOR_GRAY2RGB)

  #print("image_type: \n", type(imageLeft_rgb))
  #print("variable_type: \n", imageLeft_rgb.dtype)
  #print("shape: \n", imageLeft_rgb.shape)
  #print(imageLeft_rgb)

  #print("imageLeft: \n", imageLeft.size())
  #print("imageCenter: \n", imageCenter.size())
  #print("imageRight: \n", imageRight.size())
  #imgs = torch.cat((imageLeft, imageCenter), 0)

  '''imgs = load_images(["imageLeft.jpg", "imageCenter.jpg", "imageRight.jpg"])

  print("image_type: ", type(imgs))
  inner_sizes = [len(sublist) for sublist in imgs]
  print(f"Sizes of the inner lists: {inner_sizes}")
  print("size: ", imgs[0].size())
  print(imgs)'''

  imgs = [tensorleft, tensorcenter, tensorright]
  #print("imgs= ", type(imgs))
  #print("image_type: ", type(imgs))
  #inner_sizes = [len(sublist) for sublist in imgs]
  #print(f"Sizes of the inner lists: {inner_sizes}")
  #print("size: ", imgs[0].size())
  #print(imgs)
  #print("variable_type: \n", imgs.dtype)


  #size:
  #torch.Size([1, 3, 512, 640])
  #[tensor([[[[0.1216, 0.3373, 0.4824, ..., 0.6078, 0.6392, 0.6824],
  #device='cuda:0'), tensor([[[[0.3922, 0.4471, 0.2863,  ..., 0.2078, 0.2392, 0.2784],
  #[tensor([[[[ 29.,  90., 121.,  ..., 156., 162., 173.],
  #[101., 131., 144.,  ..., 203., 207., 201.]]]], device='cuda:0')]
  #size:
  #torch.Size([1, 3, 512, 640])

  #print(imgs[0].size())
  #print(type(imgs[0]))
  #print(imgs[0].shape)
  #print(imgs[0].is_cuda)

  #print("imgs[0] = ", {imgs[0].shape}, " imgs[1] = ", {imgs[1].shape}, " imgs[2] = ", {imgs[2].shape})
  #print("01")
  #imgs = []
  #print("02")

  #matcher = KF.LocalFeatureMatcher(KF.GFTTAffNetHardNet(10), KF.DescriptorMatcher('snn', 0.8))
  if calibration_status == 0:
    matcher = KF.LoFTR('outdoor').cuda()#pretrained='outdoor')
    IS = ImageStitcher_acg(matcher, estimator='ransac', blending_method='naive').cuda()

  with torch.inference_mode():
      out = IS(*imgs)

  out1 = out*255
  out2 = K.tensor_to_image(out1)
  out3 = out2.astype(np.uint8)

  #out3 = imageLeft

  '''out1 = K.tensor_to_image(out)
  out2 = out1*255
  out3 = out2.astype(np.uint8)'''

  #print(type(out3))
  #print(out3.shape)
  #print(out3)
  #cv2.imwrite('out3.jpg', out3)
  #out4 = cv2.cvtColor(out3, cv2.COLOR_RGB2BGR)

  image_pub = rospy.Publisher("combined_image",Image, queue_size = 10)

  try:
    #image_pub.publish(br.cv2_to_imgmsg(keypoints_all, "rgb8")) 
    #image_pub.publish(br.cv2_to_imgmsg(panorama, "unit8"))
    image_pub.publish(br.cv2_to_imgmsg(out3))
    #rospy.loginfo("published image")
  except CvBridgeError as e:
    print(e)

def main(args):
  #ci = combine_image_class()

  rospy.init_node('combined_image', anonymous=True)

  try:
    read_cameras()
    #rospy.spin()
  except KeyboardInterrupt:
    print ("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)



