import yaml
import sys
import os
import numpy as np
from typing import List, Dict

import cv2

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

def ensure_color_assigntment(traces_ids:List[int], colors:Dict[int,List[int]]) -> Dict[int,List[int]]:

    if colors is None:
        colors = {}

    for id in traces_ids:
        if not id in colors.keys():
            colors[id] = print_utils.get_random_color()

    return colors


# First, we obtain the configuration dictionary.
yaml_file = open("./config.yml", 'r')
config = yaml.load(yaml_file, Loader=yaml.FullLoader)

sys.path.insert(0, config["OBJECT_DETECTION_REPOSITORY_FOLDER"])
sys.path.insert(0, config["PYTRACKER_REPOSITORY_FOLDER"])

import object_detection.utils.read_utils as read_utils
from object_detection.object_detectors.object_detector import Object_Detector
from object_detection.print_utils import print_utils
from pytracker.tracker import Tracker

output_folder = config["OUTPUT_FOLDER"]

front_video_file_path = os.path.join(config["INPUT_FOLDER"], config["FRONT_VIDEO_NAME"])
back_video_file_path = os.path.join(config["INPUT_FOLDER"], config["BACK_VIDEO_NAME"])

front_video_gen = read_utils.generator_from_video(front_video_file_path)
back_video_gen = read_utils.generator_from_video(back_video_file_path)

object_detector = Object_Detector(config["BACKEND"], config["MODEL"], config["TRANSFORMATIONS"],config["MODEL_ORIGIN"])
print(f"Loaded object detection model.")

front_tracker = Tracker(50,5,1000)
back_tracker = Tracker(50,5,1000)

front_colors_dict = None
back_colors_dict = None
for index, (front_frame, back_frame) in enumerate(zip(front_video_gen, back_video_gen)):
    print(f"Front image {index}")
    rgb_front_frame = front_frame[:,:,[2,1,0]]
    rgb_back_frame = back_frame[:,:,[2,1,0]]

    output_front = object_detector.process_single_image(rgb_front_frame)
    print(f"Front image {index}")
    for bbox, _class, confidence in zip(output_front[0], output_front[1], output_front[2]):
        print("Bounding Box")
        print(bbox)
        print("Class")
        print(_class)
        print("Confidence")
        print(confidence)

    positions = get_positions_from_bboxes(output_front[0])
    front_track_ids = back_tracker.assign_incomming_positions(np.array(positions))
    print("Track ids")
    print(front_track_ids)
    front_colors_dict = ensure_color_assigntment(front_track_ids, front_colors_dict)

    front_drawn_image = print_utils.print_detections_on_image(output_front, front_frame)
    print(front_colors_dict)
    front_drawn_image = print_utils.print_points_on_image(positions, front_drawn_image, colors=[front_colors_dict[x] for x in front_track_ids])

    output_back = object_detector.process_single_image(rgb_back_frame)
    print(f"Back image {index}")

    for bbox, _class, confidence in zip(output_back[0], output_back[1], output_back[2]): 
        print("Bounding Box")
        print(bbox)
        print("Class")
        print(_class)
        print("Confidence")
        print(confidence)

    positions = get_positions_from_bboxes(output_back[0])
    back_track_ids = back_tracker.assign_incomming_positions(np.array(positions))
    print("Track ids")
    print(back_track_ids)
    back_colors_dict = ensure_color_assigntment(back_track_ids, back_colors_dict)

    back_drawn_image = print_utils.print_detections_on_image(output_back, back_frame)
    print(back_colors_dict)
    back_drawn_image = print_utils.print_points_on_image(positions, back_drawn_image, colors=[back_colors_dict[x] for x in back_colors_dict])

    cv2.imwrite(os.path.join(output_folder, f"front_{index}.jpg"), front_drawn_image)
    cv2.imwrite(os.path.join(output_folder, f"back_{index}.jpg"), back_drawn_image)
