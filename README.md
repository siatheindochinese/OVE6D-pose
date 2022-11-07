# OVE6D: Object Viewpoint Encoding for Depth-based 6D Object Pose Estimation (CVPR 2022
### [Original Project Page](https://dingdingcai.github.io/ove6d-pose/) | [Paper](https://arxiv.org/abs/2203.01072)


- Added real-time inferencing.

## Real-Time Inference

Real-time inference on an Intel Realsense D455 Depth Camera:
![demo_vid](assets/realtimebysia.gif)

The model follows a 2-stage pipeline:

- (1) object detection and instance segmentation using Detectron2
- (2) OVE6D pipeline (viewpoint-encoding, codebook lookup, in-plane rotation regression, centroid refinement)

While the OVE6D viewpoint encoder does not need to be retrained for new objects, the segmentation pipeline that comes before OVE6D needs to be trained for desired objects. Detectron2 is utilized due to its off-the-shelf nature, following the original authors use of Detectron2 for the LineMod dataset.

<p align="center">
    <img src ="assets/introduction_figure.png" width="500" />
</p>

## Setup
Refer to the original repository for setup instructions.

## Training for Custom Dataset
- to-be-added

## Acknowledgement
This fork uses work from [Detectron2](https://github.com/facebookresearch/detectron2) and [OVE6D](https://github.com/zju3dv/OnePose), neither of which I am originally involved in.
