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
from PIL import Image, ImageDraw
from pathlib import Path
from matplotlib import pyplot as plt
from os.path import join as pjoin

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

from lib import rendering, network
from dataset import LineMOD_Dataset
from evaluation import utils
from evaluation import config as cfg
import misc

class Dataset():
    def __init__(self, data_dir, cam_K, height, width, DEVICE='cuda'):
        self.model_dir = Path(data_dir) / 'models_eval'
        
        self.cam_K = torch.from_numpy(cam_K).to(DEVICE)
        
        self.cam_height = height
        self.cam_width = width
        
        self.model_info_file = self.model_dir / 'models_info.json'
        with open(self.model_info_file, 'r') as model_f:
            self.model_info = json.load(model_f)
            
        self.obj_model_file = dict()
        self.obj_diameter = dict()
        
        for model_file in sorted(self.model_dir.iterdir()):
            if str(model_file).endswith('.ply'):
                obj_id = int(model_file.name.split('_')[-1].split('.')[0])
                self.obj_model_file[obj_id] = model_file
                self.obj_diameter[obj_id] = self.model_info[str(obj_id)]['diameter']

parser = argparse.ArgumentParser()
parser.add_argument("--objname",
					help = ".name of obj, files must be located in /Dataspace",
					type = str,
					required = True)
parser.add_argument("--icp",
					help = ".name of obj, files must be located in /Dataspace",
					type = bool,
					default = False)
args = parser.parse_args()

DEVICE = 'cuda'
objname = args.objname
use_icp = args.icp
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
	spatial = rs.spatial_filter()
	hole_filling = rs.hole_filling_filter(0)
	colorizer = rs.colorizer()
	
	# init opencv window
	cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('frame', width * 2, height * 2)
	cv2.namedWindow('depth',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('depth', width * 2, height * 2)
	
	# init video recorder
	#writer = cv2.VideoWriter('basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width,height))
	
	#################################
	# load detectron maskrcnn model #
	#################################
	rcnnIdx_to_datasetIds_dict = {0:1} # modify for custom dataset
	rcnnIdx_to_datasetCats_dict = {0:objname} # modify for custom dataset
	
	rcnn_cfg = get_cfg()
	rcnn_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
	rcnn_cfg.MODEL.WEIGHTS = os.path.abspath(pjoin('checkpoints', objname+'.pth'))
	rcnn_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
	rcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1 # the predicted category scores
	predictor = DefaultPredictor(rcnn_cfg)
	print('Mask-RCNN has been loaded!')
	
	##############
	# load OVE6D #
	##############
	ckpt_file = pjoin('checkpoints', "OVE6D_pose_model.pth")
	model_net = network.OVE6D().to(DEVICE)
	model_net.load_state_dict(torch.load(ckpt_file))
	model_net.eval()
	
	print('OVE6D has been loaded!')
	
	#########################
	# load abstract Dataset #
	#########################
	eval_dataset = Dataset(pjoin('Dataspace', objname), K_full, 480, 640)
	cfg.DATASET_NAME = objname
	tar_obj_id = 1
	tar_rcnn_d = tar_obj_id - 1
	obj_name = rcnnIdx_to_datasetCats_dict[tar_rcnn_d]
	
	############################
	# load viewpoint codebooks #
	############################
	cfg.VIEWBOOK_BATCHSIZE = 200
	codebook_saving_dir = pjoin('evaluation/object_codebooks',
								cfg.DATASET_NAME, 
								'zoom_{}'.format(cfg.ZOOM_DIST_FACTOR), 
								'views_{}'.format(str(cfg.RENDER_NUM_VIEWS))) # modify for custom dataset
	
	object_codebooks = utils.OVE6D_codebook_generation(codebook_dir=codebook_saving_dir, 
													   model_func=model_net,
													   dataset=eval_dataset,
													   config=cfg,
													   device=DEVICE)
	print('Object codebooks have been loaded!')
	
	############################
	# target object properties #
	############################
	tar_obj_codebook = object_codebooks[tar_obj_id]
	obj_mesh = tar_obj_codebook['obj_mesh']
	obj_diameter = tar_obj_codebook['diameter']
	corner_pts = obj_mesh.bounds  # object bounding box vertices
	obj_pcl = obj_mesh.vertices
	N_pcl = len(obj_pcl)
	select_obj_idxes = torch.randperm(N_pcl)[:500]
	select_obj_pts = torch.tensor(obj_pcl[select_obj_idxes])
	
	#####################################
	# adjust OVE6D inference parameters #
	#####################################
	cfg.VP_NUM_TOPK = 50   # the retrieval number of viewpoint 
	cfg.RANK_NUM_TOPK = 5  # the ranking number of full 3D orientation
	cfg.USE_ICP = use_icp
	
	#################
	# load renderer #
	#################
	obj_renderer = rendering.Renderer(width=cfg.RENDER_WIDTH, height=cfg.RENDER_HEIGHT)
	
	while True:
		print('new frame')
		print('')
		## stream the next frame ##
		frameset = pipe.wait_for_frames()
		frameset = align.process(frameset)
		color_frame = frameset.get_color_frame()
		depth_frame = frameset.get_depth_frame()
		filtered_depth = spatial.process(depth_frame)
		dpt = np.asanyarray(filtered_depth.get_data()) * depth_scale
		bgr = np.asanyarray(color_frame.get_data())
		
		view_depth = torch.tensor(dpt).to(DEVICE)
		
		## object segmentation ##
		output = predictor(bgr)
		rcnn_pred_ids = output["instances"].pred_classes
		rcnn_pred_masks = output["instances"].pred_masks
		rcnn_pred_scores = output["instances"].scores
		obj_detected = (len(rcnn_pred_ids != 0))
		
		'''
		v = Visualizer(bgr[:, :, ::-1], 
					   scale=1.0, 
					   instance_mode=ColorMode.IMAGE_BW)
		result = v.draw_instance_predictions(output["instances"].to("cpu")).get_image()[:, :, ::-1]
		'''
		
		## OVE6D pipeline (if object detected) ##
		if obj_detected:
			obj_masks = rcnn_pred_masks # NxHxW
			obj_depths = view_depth[None, ...] * obj_masks
			tar_obj_depths = obj_depths[tar_rcnn_d==rcnn_pred_ids]
			tar_obj_masks = rcnn_pred_masks[tar_rcnn_d==rcnn_pred_ids]
			tar_obj_scores = rcnn_pred_scores[tar_rcnn_d==rcnn_pred_ids]
			
			mask_pixel_count = tar_obj_masks.view(tar_obj_masks.size(0), -1).sum(dim=1)
			valid_idx = (mask_pixel_count >= 100)
			if valid_idx.sum() == 0:
				mask_visib_ratio = mask_pixel_count / mask_pixel_count.max()
				valid_idx = mask_visib_ratio >= 0.05
			
			tar_obj_masks = tar_obj_masks[valid_idx] # select the target object instance masks
			tar_obj_depths = tar_obj_depths[valid_idx]
			tar_obj_scores = tar_obj_scores[valid_idx]
			
			pose_ret, rcnn_idx = utils.OVE6D_rcnn_full_pose(model_func=model_net, 
															obj_depths=tar_obj_depths,
															obj_masks=tar_obj_masks,
															obj_rcnn_scores=tar_obj_scores,
															obj_codebook=tar_obj_codebook, 
															cam_K=eval_dataset.cam_K,
															config=cfg, 
															device=DEVICE,
															obj_renderer=obj_renderer, 
															return_rcnn_idx=True)

			if cfg.USE_ICP:
				pose_R = pose_ret['icp1_R'] # with ICP after pose selection
				pose_t = pose_ret['icp1_t'] # with ICP after pose selection
			else:
				pose_R = pose_ret['raw_R'] # without ICP
				pose_t = pose_ret['raw_t'] # without ICP
			
		if obj_detected:
			PD_pose = torch.eye(4, dtype=torch.float32)
			PD_pose[:3, 3] = pose_t
			PD_pose[:3, :3] = cfg.POSE_TO_BOP(pose_R)
			
			PD_2D_bbox = misc.box_2D_shape(points=corner_pts,
										   pose=PD_pose,
										   K=eval_dataset.cam_K)
			
			PD_shape = misc.bbox_to_shape(PD_2D_bbox.tolist())
			bbox_img = Image.fromarray(bgr)
			draw = ImageDraw.Draw(bbox_img)
			draw.line(PD_shape, (0, 255, 0), 3)
			result = np.array(bbox_img)
			
		else:
			result = bgr
		
		# display processed frame
		colorized_depth = np.asanyarray(colorizer.colorize(filtered_depth).get_data())
		cv2.imshow('depth', colorized_depth)
		cv2.imshow('frame', result)
		#writer.write(result)
		print('')
		print('')
		
		# detect 'q' key to exit the loop
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
			
	# detect 'q' key to exit the loop
	pipe.stop()
	cv2.destroyAllWindows()
	#writer.release()
	
if __name__ == '__main__':
	main()
