import yaml
import sys
import os
import numpy as np
from typing import List

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

for index, (front_frame, back_frame) in enumerate(zip(front_video_gen, back_video_gen)):
    print(index)
    rgb_front_frame = front_frame[:,:,[2,1,0]]
    rgb_back_frame = back_frame[:,:,[2,1,0]]
    output_front = object_detector.process_single_image(rgb_front_frame)
    print(f"Front image {index}")
    for bbox, _class, confidence in zip(output_front[0], output_front[1], output_front[2]):

        positions = get_positions_from_bboxes(bbox)
        front_track_ids = front_tracker.assign_incomming_positions(np.array(positions))
        print("Track ids")
        print(front_track_ids)
        print("Bounding Box")
        print(bbox)
        print("Class")
        print(_class)
        print("Confidence")
        print(confidence)
    output_back = object_detector.process_single_image(rgb_back_frame)
    print(f"Back image {index}")
    for bbox, _class, confidence in zip(output_back[0], output_back[1], output_back[2]): 

        positions = get_positions_from_bboxes(bbox)
        back_track_ids = back_tracker.assign_incomming_positions(np.array(positions))
        print("Track ids")
        print(back_track_ids)
        print("Bounding Box")
        print(bbox)
        print("Class")
        print(_class)
        print("Confidence")
        print(confidence)

    front_drawn_image = print_utils.print_detections_on_image(output_front, front_frame)
    back_drawn_image = print_utils.print_detections_on_image(output_back, back_frame)

    cv2.imwrite(os.path.join(output_folder, f"front_{index}.jpg"), front_drawn_image)
    cv2.imwrite(os.path.join(output_folder, f"back_{index}.jpg"), back_drawn_image)
