# stitching

ROS Noetic package for stitching image feeds from multiple FLIR USB cameras (left/center/right) into a single panorama, using the [`flir_cameras`](https://github.com/viron11111/flir_cameras) driver package for camera input.

## Contents

- `src/combine*.py` — successive iterations of the stitching pipeline (feature matching, seam masking, cropping/blending). `combine_6A.py` is the most recent iteration.
- `src/stitching_combine.cpp` — C++ implementation of the stitching node.
- `src/image_feature_matching_test.py`, `src/high_speed_edge_detection*.py` — supporting experiments for feature matching and edge detection.
- `images/` — sample input frames (`flir_left_image.png`, `flir_center_image.png`, `flir_right_image.png`) and an example stitched output (`stitched_result.jpg`).
- `sirrus_lan_files/` — host-specific launch/network config (`the_works.launch`, `hosts`, `.bashrc`) for running this on the `sirrus_lan` rig.
- `rotation_matrices_tensors` — saved camera rotation/calibration data used by the stitching pipeline.

## Usage

Requires `flir_cameras` (or equivalent camera driver publishing left/center/right image topics) running first. See `sirrus_lan_files/the_works.launch` for a full multi-camera + stitching launch example.

## Provenance

Recovered from a ROS Noetic workspace (Ubuntu 20.04, both EOL) in 2026-07 and pushed here for safekeeping/versioning; not previously under version control.
