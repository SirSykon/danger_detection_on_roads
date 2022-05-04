"""
Author: Jorge García <jrggcgz@gmail.com>


"""
import numpy as np
import sys
import os
from typing import List, Dict, Tuple

"""
if not config["OBJECT_DETECTION_REPOSITORY_FOLDER"] in sys.path:
    sys.path.insert(0, config["OBJECT_DETECTION_REPOSITORY_FOLDER"])

if not config["PYTRACKER_REPOSITORY_FOLDER"] in sys.path:
    sys.path.insert(0, config["PYTRACKER_REPOSITORY_FOLDER"])
"""
from object_detection.object_detectors.object_detector import Object_Detector
from object_detection.print_utils import print_utils

from pytracker.tracker import Tracker
from my_utils import get_positions_from_bboxes, ensure_color_assigntment

class Camera_Information_Register:
    """
    Class to handle a register of information obtained by a video using a tracker over an object detector.
    """

    def __init__(self, config:Dict, object_detector:Object_Detector, output_folder:str = None) -> None:
        """
        config:Dict -> Dictionary with the configuration parameters.
        object_detector:Object_Detector -> Object instance to get object detection from image.
        output_folder:str -> Folder to save outputs.
        """

        self.tracker = Tracker(config["MAXIMUM_DISTANCE_TO_ASSOCIATE"], config["MAXIMUM_FRAMES_TO_SKIP"], config["VALUE_TO_USE_AS_INF"])
        self.information = {}
        self.colors = {}
        self.object_detector = object_detector
        self.print_debug_info = config["PRINT_DEBUG_INFO"]
        self.fps = config["FPS"]
        self.output_folder = output_folder
        print("Register Initialized.")

    def save_information(self, track_id:int, bbox:List[int], class_:int, confidence:float, time:int, image:np.ndarray) -> None:
        """
        Abstract method to obtain the detected object related information.
        track_id:int -> object tracker identification.
        bbox:List[int] -> bbox assumed as [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
        class_:int -> Class value.
        confidence:float -> Value in [0,1] with the class_ confidence.
        time:int -> Time value of the image containing the object.
        image:np.ndarray -> RGB image as numpy matrix.
        """

        bbox = data[0]
        class_ = data[1]
        confidence = data[2]

        if not track_id in self.information.keys():
            self.information[track_id] = []
        track_id_register = self.information[track_id]
        track_id_register.append(info)
        self.information[track_id] = register

        return None

    def plot_information(self, output, track_id:int = None, track_length_to_print:int = None) -> None:
        """
        Plot information from the register.
        output:str -> Folder to save plots in.
        track_id:int -> object tracker identification. If track_id is None, information about all track_id will be plotted, otherwise only information about track_id will be plotted.
        track_length_to_print:int -> Minimum length of a track to get info from it.
        """

        if track_id is None:
            for t_id in self.information.keys():
                info = self.information[t_id]
                if track_length_to_print is None or len(info) >= track_length_to_print:
                    self.plot_aux_information(t_id, info, output)
        else:
            info = self.information[track_id]
            if track_length_to_print is None or len(info) >= track_length_to_print:
                self.plot_aux_information(track_id, info, output)

    def plot_aux_information(self, track_id:int, information:List, output:str) -> None:
        """
        Abstract method to be overwritten to plot register information.
        track_id:int -> object tracker identification.
        information:List -> Generic list with the information to be plotted.
        output:str -> Folder to save plots in.
        """

        raise(NotImplementedError)

    def plot_history_on_image(self, track_id:int, rgb_frame:np.ndarray, output_path:str) -> None:
        """
        Abstract method to be overwritten to plot register information.
        track_id:int -> object tracker identification.
        rgb_frame:np.ndarray -> RGB image as numpy matrix.
        output:str -> Folder to save plots in.
        """

        raise(NotImplementedError)

    def process_image(self, rgb_frame:np.ndarray, time:int, classes_to_use:List[int] = None) -> np.ndarray:
        """
        Method to process each image. Object from the image will obe obtained using self.object_detector, then their bbox positions will be associated by self.tracker to track the objects.
        method self.get_information (tipically overwritten by a child class) will save the object information related to its track id.
        self.colors will be used to print objects with track-related colors.
        rgb_frame:np.ndarray -> RGB image as numpy matrix.
        time:int -> Time value of the image containing the object.
        classes_to_filter:List[int] -> All detected objects with class not included in the list will be ignored. If None, no class wil be ignored.
        """

        output = self.object_detector.process_single_image(rgb_frame)

        bboxes = output[0]
        classes = output[1]
        confidences = output[2]
        
        if self.print_debug_info:
            for bbox, class_, confidence in zip(bboxes, classes, confidences):        
                if classes_to_use is None or class_ in classes_to_use:
                    print("Bounding Box")
                    print(bbox)
                    print("Class")
                    print(class_)
                    print("Confidence")
                    print(confidence)

        # We get positions in order to get track ids.
        positions = get_positions_from_bboxes(bboxes)
        track_ids = self.tracker.assign_incomming_positions(np.array(positions))
        self.colors = ensure_color_assigntment(track_ids, self.colors)

        infos = []
        for track_id, bbox, class_, confidence in zip(track_ids,bboxes,classes,confidences):
            if classes_to_use is None or class_ in classes_to_use:
                self.save_information(track_id, bbox, class_, confidence, time, rgb_frame)
                self.plot_history_on_image(track_id, rgb_frame, os.path.join(self.output_folder, "history"))

        if self.print_debug_info:

            drawn_image = print_utils.print_detections_on_image(output, rgb_frame[:,:,[2,1,0]])

            drawn_image = print_utils.print_points_on_image(positions, drawn_image, colors=[self.colors[x] for x in track_ids])
            drawn_image = print_utils.print_info_on_image(infos, positions, drawn_image, colors=[self.colors[x] for x in track_ids])

            return drawn_image

        else:
            return None
