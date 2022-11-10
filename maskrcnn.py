import os
import cv2
import sys
import json
import math
import time
import torch
import argparse
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
from matplotlib import pyplot as plt
from os.path import join as pjoin

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument("--weights",
					help = ".pth file located in /checkpoints",
					type = str,
					required = True)
args = parser.parse_args()

DEVICE = 'cuda'
def main():
	##################################################
	# initialize realsense and opencv configurations #
	##################################################
	# init realsense camera
	pipe = rs.pipeline()
	rscfg = rs.config()
	width, height = 640, 480
	rscfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)
	rscfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 60)
	profile = pipe.start(rscfg)
	
	# getting realsense depth attributes
	depth_sensor = profile.get_device().first_depth_sensor()
	depth_sensor.set_option(rs.option.laser_power,360)
	depth_scale = depth_sensor.get_depth_scale()
	clipping_distance_in_meters = 1
	clipping_distance = clipping_distance_in_meters / depth_scale
	
	# getting realsense intrinsics
	intrin = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
	K_full = np.array([[intrin.fx, 0, intrin.ppx],
					   [0, intrin.fy, intrin.ppy],
					   [0,         0,          1]])
	
	# realsense utility functions
	align = rs.align(rs.stream.color)
	colorizer = rs.colorizer()
	
	# init opencv window
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', width, height)
	
	# init video recorder
	#writer = cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width,height))
	
	#################################
	# load detectron maskrcnn model #
	#################################
	rcnn_cfg = get_cfg()
	rcnn_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	rcnn_cfg.MODEL.WEIGHTS = os.path.abspath(pjoin('checkpoints', args.weights))
	rcnn_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
	rcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01 # the predicted category scores
	predictor = DefaultPredictor(rcnn_cfg)
	print('Mask-RCNN has been loaded!')
	
	while True:
		## stream the next frame ##
		frameset = pipe.wait_for_frames()
		frameset = align.process(frameset)
		color_frame = frameset.get_color_frame()
		depth_frame = frameset.get_depth_frame()
		dpt = np.asanyarray(depth_frame.get_data()) * depth_scale
		bgr = np.asanyarray(color_frame.get_data())
		
		## instance segmentation ##
		outputs = predictor(bgr)
		print('instances: ', len(outputs['instances']))
		obj_detected = (len(outputs['instances']) != 0)
		
		if obj_detected:
			v = Visualizer(bgr[:, :, ::-1],
						   scale=1.0,
						   instance_mode=ColorMode.IMAGE_BW)
			out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
			result = out.get_image()[:, :, ::-1]
		else:
			result = bgr
		
		## display processed frame ##
		cv2.imshow('frame', result)
		#writer.write(result)
		
		## detect 'q' key to exit the loop ##
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
	pipe.stop()
	cv2.destroyAllWindows()
	#writer.release()
	
if __name__ == '__main__':
	main()
