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
from register import Camera_Information_Register


def analyze_information_tuple_sequence(tuple_sequence:List[Tuple[int,int,int]], fps:int, prediction_time_range:int = 3, output_graph_name:str = None) -> None:
    """
    Function to analyze a sequence of tuples. Each tuple structure is (x,y,area) with (x,y) as the position of the vehicle and area the bbox area.
    We will analyze each tuple as a position in a 3D space related to a time.
    """

    sequence_length = len(tuple_sequence)
    sequence_times = np.arange(0, sequence_length/fps, 1/fps)
    reg = LinearRegression().fit(sequence_times, np.array(tuple_sequence))

    prediction_time = np.arange(sequence_times[-1], sequence_times[-1]+prediction_time_range, 1/fps)
    pred = reg.predict(prediction_time)
    if not output_graph_name is None:
        fig = plot_tuple_sequence(tuple_sequence, pred, f"{os.path.basename(output_graph_name)}")
        plt.set_current_figure(fig)
        plt.savefig(output_graph_name)

def analyze_information_dict(information_dict:Dict[int,List[Tuple[int,int,int]]], frame_index:int, fps:int, prediction_time_range:int = 3, output_analysis_folder:str = None)-> None:
    for key in information_dict.keys():
        information_register = information_dict[key]
        analyze_information_tuple_sequence(information_register, fps, prediction_time_range, os.path.join(output_analysis_folder,f"frame_{frame_index}_track_{key}.jpg"))

# Set folders.
output_folder = config["OUTPUT_FOLDER"]
output_analysis_folder_back = os.path.join(output_folder, "analysis_back")
output_analysis_folder_front = os.path.join(output_folder, "analysis_front")

front_video_file_path = os.path.join(config["INPUT_FOLDER"], config["FRONT_VIDEO_NAME"])
back_video_file_path = os.path.join(config["INPUT_FOLDER"], config["BACK_VIDEO_NAME"])

# Input generators.
front_video_gen = read_utils.generator_from_video(front_video_file_path)
back_video_gen = read_utils.generator_from_video(back_video_file_path)

# Object detection initialization.
object_detector = Object_Detector(config["BACKEND"], config["MODEL"], config["TRANSFORMATIONS"],config["MODEL_ORIGIN"])
print(f"Loaded object detection model.")

front_camera_register = Camera_Information_Register(config, object_detector)
back_camera_register = Camera_Information_Register(config, object_detector)

# Data initialization.
back_out = None                 # Back output video.
front_out = None                # Front input video.

for index, (front_frame, back_frame) in enumerate(zip(front_video_gen, back_video_gen)):
    print(f"Image {index}")
    # BGR to RGB.
    rgb_front_frame = front_frame[:,:,[2,1,0]]  
    rgb_back_frame = back_frame[:,:,[2,1,0]]

    front_drawn_image = front_camera_register.process_image(rgb_front_frame)
    analyze_information_dict(front_information_dict, index, 30, 3, output_analysis_folder = output_analysis_folder_back)

    back_drawn_image = front_camera_register.process_image(rgb_back_frame)
    analyze_information_dict(front_information_dict, index, 30, 3, output_analysis_folder = output_analysis_folder_front)

    if config["SAVE_OUTPUT_VIDEO"] and back_out is None:
        # If there is no back output video, we initialize it.
        height, width, channels = front_frame.shape
        size = (height,width)
        back_out = cv2.VideoWriter('./output/back.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        
    back_out.write(back_drawn_image)

    if config["SAVE_OUTPUT_VIDEO"] and front_out is None:
        # If there is no front output video, we initialize it.
        height, width, channels = back_frame.shape
        size = (height,width)
        front_out = cv2.VideoWriter('./output/front.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    front_out.write(front_drawn_image)
    
    if config["SAVE_OUTPUT_VIDEO"]:
        cv2.imwrite(os.path.join(output_folder, f"front_{index}.jpg"), front_drawn_image)
        cv2.imwrite(os.path.join(output_folder, f"back_{index}.jpg"), back_drawn_image)

back_out.release()
front_out.release()
print(front_information_dict)
print(back_information_dict)
