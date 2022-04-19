"""
Author: Jorge García <jrggcgz@gmail.com>


"""
import numpy as np
import sys

if not config["OBJECT_DETECTION_REPOSITORY_FOLDER"] in sys.path:
    sys.path.insert(0, config["OBJECT_DETECTION_REPOSITORY_FOLDER"])

if not config["PYTRACKER_REPOSITORY_FOLDER"] in sys.path:
    sys.path.insert(0, config["PYTRACKER_REPOSITORY_FOLDER"])

from object_detection.object_detectors.object_detector import Object_Detector
from utils import get_positions_from_bboxes, ensure_color_assigntment, get_area_from_bbox, get_center_from_bbox, plot_tuple_sequence
from pytracker.tracker import Tracker

class Camera_Information_Register:
    """
    Class to handle a register of information obtained by a video using a tracker over an object detector.
    """

    def __init__(self, config, object_detector) -> None:
        self.tracker = Tracker(config["MAXIMUM_DISTANCE_TO_ASSOCIATE"], config["MAXIMUM_FRAMES_TO_SKIP"], config["VALUE_TO_USE_AS_INF"])
        self.informaion = {}
        self.colors = {}
        self.object_detector = object_detector
        self.print_debug_info = config["PRINT_DEBUG_INFO"]
        print("Register Initialized.")

    def process_image(rgb_frame:np.ndarray) -> np.ndarray:

        output = self.object_detector.process_single_image(rgb_frame)

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
        track_ids = self.tracker.assign_incomming_positions(np.array(positions))

        infos = []
        for track_id, bbox in zip(track_ids,bboxes):
            area = get_area_from_bbox(bbox)
            center = get_center_from_bbox(bbox)
            info = (center[0], center[1], area)
            infos.append(str(info))
            if not track_id in self.information.keys():
                self.information[track_id] = []
            register = self.information[track_id]
            register.append(info)
            self.information[track_id] = register

        if print_debug_info:
            print("Track ids")
            print(track_ids)
            self.colors = ensure_color_assigntment(track_ids, self.colors)

            drawn_image = print_utils.print_detections_on_image(output, rgb_frame[:,:,[2,1,0]])

            print("Colors dict")
            print(self.colors)

            drawn_image = print_utils.print_points_on_image(positions, drawn_image, colors=[self.colors[x] for x in track_ids])
            drawn_image = print_utils.print_info_on_image(infos, positions, drawn_image, colors=[self.colors[x] for x in track_ids])

            return drawn_image

        else:
            return None
