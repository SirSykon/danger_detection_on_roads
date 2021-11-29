import yaml
import sys
import os

import cv2

# First, we obtain the configuration dictionary.
yaml_file = open("./config.yml", 'r')
config = yaml.load(yaml_file, Loader=yaml.FullLoader)

sys.path.insert(0, config["OBJECT_DETECTION_REPOSITORY_FOLDER"])

import object_detection.utils.read_utils as read_utils
from object_detection.object_detectors.object_detector import Object_Detector
from object_detection.print_utils import print_utils

output_folder = config["OUTPUT_FOLDER"]

front_video_file_path = os.path.join(config["INPUT_FOLDER"], config["FRONT_VIDEO_NAME"])
back_video_file_path = os.path.join(config["INPUT_FOLDER"], config["BACK_VIDEO_NAME"])

front_video_gen = read_utils.generator_from_video(front_video_file_path)
back_video_gen = read_utils.generator_from_video(back_video_file_path)

object_detector = Object_Detector(config["BACKEND"], config["MODEL"], config["TRANSFORMATIONS"],config["MODEL_ORIGIN"])
print(f"Loaded object detection model.")

for index, (front_frame, back_frame) in enumerate(zip(front_video_gen, back_video_gen)):
    print(index)
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
    output_back = object_detector.process_single_image(rgb_back_frame)
    print(f"Back image {index}")
    for bbox, _class, confidence in zip(output_back[0], output_back[1], output_back[2]): 
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
