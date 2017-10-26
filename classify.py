import models.vgg16 as vgg16
import numpy as np
import tensorflow as tf
import argparse
import misc.utils as utils
import models.vgg_utils as vgg_utils
import os
gpu_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id
def run(image_filename, class_label):
	grad_CAM_map= utils.grad_CAM_plus(image_filename, class_label)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--class_label', default=-1, type=int, help='if -1 (default) choose predicted class, else user supplied int')
	parser.add_argument('-gpu', '--gpu_device', default=0, type=str, help='if 0 (default) choose gpu 0, else user supplied int')
	parser.add_argument('-f', '--file_name', type=str, help="Give filename on whose Grad-CAM++ visualization you wish to see, please ensure the given image file is location into the images/ subdirectory")
	args = parser.parse_args()
	global gpu_id
	gpu_id = args.gpu_device
	run(args.file_name, args.class_label)

if __name__ == '__main__':
    main()
