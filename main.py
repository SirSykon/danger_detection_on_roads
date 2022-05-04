"""
Author: Jorge García <jrggcgz@gmail.com>

"""

import yaml
import sys
import os
import numpy as np
from typing import List, Dict, Tuple
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import cv2

# First, we obtain the configuration dictionary.
yaml_file = open("./config.yml", 'r')
config = yaml.load(yaml_file, Loader=yaml.FullLoader)

sys.path.insert(0, config["OBJECT_DETECTION_REPOSITORY_FOLDER"])
sys.path.insert(0, config["PYTRACKER_REPOSITORY_FOLDER"])

import object_detection.utils.read_utils as read_utils
from object_detection.object_detectors.object_detector import Object_Detector
from area_position_register import Area_Position_Register as Register

# Set folders.
output_folder = os.path.join(config["OUTPUT_FOLDER"], config["VIDEO_NAME"])
output_analysis_folder = os.path.join(output_folder, "analysis")
output_images_folder = os.path.join(output_folder, "images")
output_danger_values_file = os.path.join(output_folder, "danger_values.npy")

if not os.path.isdir(output_analysis_folder):
    os.makedirs(output_analysis_folder)

if not os.path.isdir(output_images_folder):
    os.makedirs(output_images_folder)

video_file_path = os.path.join(config["INPUT_FOLDER"], config["VIDEO_NAME"])

# Input generators.
video_gen = read_utils.generator_from_video(video_file_path)

# Object detection initialization.
object_detector = Object_Detector(config["BACKEND"], config["MODEL"], config["TRANSFORMATIONS"],config["MODEL_ORIGIN"])
print(f"Loaded object detection model.")

camera_register = Register(config, object_detector, output_folder)

# Data initialization.
out_video = None                # Output video.

for index, frame in enumerate(video_gen):
    print(f"Image {index}")
    # BGR to RGB.
    rgb_frame = frame[:,:,[2,1,0]]  

    drawn_image = camera_register.process_image(rgb_frame, index/config["FPS"], classes_to_use = config["CLASSES_TO_USE"])

    #register.analyze_information_dict(front_information_dict, index, 30, 3, output_analysis_folder = output_analysis_folder_back)

    if config["SAVE_OUTPUT_VIDEO"] and out_video is None:
        # If there is no front output video, we initialize it.
        height, width, channels = rgb_frame.shape
        size = (height,width)
        out_video = cv2.VideoWriter('./output/video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)


    if config["SAVE_OUTPUT_VIDEO"]:
        out_video.write(drawn_image)
    
    if config["SAVE_OUTPUT_IMAGES"]:
        cv2.imwrite(os.path.join(output_folder, f"{index} (time {index/config['FPS']}.jpg"), drawn_image)

camera_register.plot_information(output_images_folder, track_length_to_print = config["TRACK_LENGTH_TO_CALCULATE_DANGER"])
danger_values = camera_register.get_all_danger_values()
np.save(output_danger_values_file, danger_values)

if config["SAVE_OUTPUT_VIDEO"]:
    out_video.release()
