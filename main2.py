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
from object_detection.print_utils import print_utils
from pytracker.tracker import Tracker

def get_positions_from_bboxes(list_of_bbox:List[List[int]], position_process:str="average") -> List[int]:
    """
    list_of_bbox:List[List[int]] -> Each bbox is assumed as [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
    position_process:str -> Way to process the information to get the position. Options are "average". Default "average".
    """
    list_of_positions = []
    for bbox in list_of_bbox:
        [x,y,width,height] = bbox
        if position_process == "average":
            list_of_positions.append([x+width//2,y+height//2])

    return list_of_positions

def ensure_color_assigntment(tracks_ids:List[int], colors:Dict[int,List[int]]) -> Dict[int,List[int]]:
    """
    Function to ensure each track id has an asssociated color. Otherwise, we generate a random color.
    tracks_ids:List[int] -> List of tracks ids.
    
    """
    if colors is None:
        colors = {}

    for id in tracks_ids:
        if not id in colors.keys():
            colors[id] = print_utils.get_random_color()

    return colors

def get_area_from_bbox(bbox:List[int])->int:
    """
    Function to get a bbox and return the number of pixels it contains.
    bbox:List[int] -> bbox must have a shape [x,y, width, height].
    """
    area = bbox[2]*bbox[3]
    return area

def get_center_from_bbox(bbox:List[int])->int:
    """
    Function to get a bbox and return its center.
    bbox:List[int] -> bbox must have a shape [x,y, width, height].
    """
    [x,y, width, height] = bbox
    center = (x+width/2, y+height/2)
    return center

def process_image(rgb_frame:np.ndarray, tracker:Tracker, information_dict:Dict[int,List[Tuple[int,int,int]]], colors_dict:Dict[int,List[int]], print_debug_info = config["PRINT_DEBUG_INFO"]) -> None:

    output = object_detector.process_single_image(rgb_frame)

    bboxes = output[0]
    classes = output[1]
    confidences = output[2]
    
    if print_debug_info:
        for bbox, _class, confidence in zip(bboxes, classes, confidences):        
            print("Bounding Box")
            print(bbox)
            print("Class")
            print(_class)
            print("Confidence")
            print(confidence)

    # We get positions in order to get track ids.
    positions = get_positions_from_bboxes(bboxes)
    track_ids = tracker.assign_incomming_positions(np.array(positions))

    infos = []
    for track_id, bbox in zip(track_ids,bboxes):
        area = get_area_from_bbox(bbox)
        center = get_center_from_bbox(bbox)
        info = (center[0], center[1], area)
        infos.append(str(info))
        if not track_id in information_dict.keys():
            information_dict[track_id] = []
        register = information_dict[track_id]
        register.append(info)
        information_dict[track_id] = register

    if print_debug_info:
        print("Track ids")
        print(track_ids)
    colors_dict = ensure_color_assigntment(track_ids, colors_dict)

    drawn_image = print_utils.print_detections_on_image(output, rgb_frame[:,:,[2,1,0]])

    if print_debug_info:
        print("Colors dict")
        print(colors_dict)

    drawn_image = print_utils.print_points_on_image(positions, drawn_image, colors=[colors_dict[x] for x in track_ids])
    drawn_image = print_utils.print_info_on_image(infos, positions, drawn_image, colors=[colors_dict[x] for x in track_ids])

    return drawn_image, tracker, information_dict, colors_dict

def plot_tuple_sequence(tuple_sequence:List[Tuple[int,int,int]], prediction_sequence:np.ndarray, plot_name:str, color:List[int]=[128.,128.,128.]) -> None:
    sequence_length = len(tuple_sequence)
    numpy_sequence = np.array(tuple_sequence)
    prediction_length = prediction_sequence.shape[0]
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    fig.suptitle(plot_name)
    ax1.plot(range(sequence_length), numpy_sequence[:,0], c=color)
    ax2.plot(range(sequence_length), numpy_sequence[:,1], c=color)
    ax3.plot(range(sequence_length), numpy_sequence[:,2], c=color)

    ax1.plot(range(sequence_length,sequence_length+prediction_length), prediction_sequence[:,0], c='r')
    ax2.plot(range(sequence_length,sequence_length+prediction_length), prediction_sequence[:,1], c='r')
    ax3.plot(range(sequence_length,sequence_length+prediction_length), prediction_sequence[:,2], c='r')

    return fig

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

# Tracker initialization.
front_tracker = Tracker(50,5,1000)
back_tracker = Tracker(50,5,1000)

# Data initialization.
front_information_dict = {}     # Dictionary with front information.
back_information_dict = {}      # Dictionary with back information.
front_colors_dict = None        # Dictionary with front colors to plot.
back_colors_dict = None         # Dictionary with back colors to plot.
back_out = None                 # Back output video.
front_out = None                # Front input video.

for index, (front_frame, back_frame) in enumerate(zip(front_video_gen, back_video_gen)):
    print(f"Image {index}")
    # BGR to RGB.
    rgb_front_frame = front_frame[:,:,[2,1,0]]  
    rgb_back_frame = back_frame[:,:,[2,1,0]]

    front_drawn_image, front_tracker, front_information_dict, front_colors_dict = process_image(rgb_front_frame, front_tracker, front_information_dict, front_colors_dict)
    analyze_information_dict(front_information_dict, index, 30, 3, output_analysis_folder = output_analysis_folder_back)

    back_drawn_image, back_tracker, back_information_dict, back_colors_dict = process_image(rgb_back_frame, back_tracker, back_information_dict, back_colors_dict)
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
