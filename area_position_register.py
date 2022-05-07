"""
Author: Jorge García <jrggcgz@gmail.com>


"""
import cv2
import math
import os
import random
import numpy as np
from typing import List, Dict, Tuple
from register import Camera_Information_Register
from my_utils import get_area_from_bbox, get_center_from_bbox, plot_tuple_sequence
from object_detection.object_detectors.object_detector import Object_Detector
from sklearn.linear_model import LinearRegression, RANSACRegressor
import matplotlib.pyplot as plt

class Area_Position_Register(Camera_Information_Register):
    """
    Class to handle a register of area and position based information related to each object obtained from a video using a tracker over an object detector.
    """

    def __init__(self, config:Dict, object_detector:Object_Detector, output_folder:str = None) -> None:
        """
        config:Dict -> Dictionary with the configuration parameters.
        object_detector:Object_Detector -> Object instance to get object detection from image.
        output_folder:str -> Folder to save outputs.
        """

        self.values_to_calculate_danger = config["VALUES_TO_CALCULATE_DANGER"]
        self.track_length_to_calculate_danger = config["TRACK_LENGTH_TO_CALCULATE_DANGER"]
        self.danger_mode = config["DANGER_MODE"]
        self.relevance_threshold = config["RELEVANCE_THRESHOLD"]
        self.relevant_only = config["RELEVANCE_FILTER"]
        self.all_danger_values = []
        self.videos_to_get_danger_threshold = config["VIDEOS_TO_GET_DANGER_THRESHOLD"]
        self.percentile_to_get_danger_threshold = config["THRESHOLD_PERCENTILE"]
        self.explicit_threshold = config["EXPLICIT_THRESHOLD"]
        self.number_of_skipped_prints_when_plotting_history_on_images = config["NUMBER_OF_SKIPPED_PRINTS_WWHEN_PLOTTING_HISTORY_ON_IMAGES"]
        if self.danger_mode == "ransac":
            self.residual_threshold = config["RANSAC_RESIDUAL_THRESHOLD"]
        
        super().__init__(config, object_detector, output_folder)

        self.danger_threshold = self.get_danger_threshold()
        print(f"Danger Threshold: {self.danger_threshold}")

    def calculate_danger(self,values_in:np.ndarray, times:np.ndarray):
        """
        Method to get a danger value based on the slope of a linear regression over values.
        values_in:np.ndarray -> Raw values to work with.
        times:np.ndarray -> times associated to each of the values.

        """
        values = self.turn_to_line(values_in)
        if self.danger_mode == "linear_regression":
            reg = LinearRegression().fit(times[-1*self.values_to_calculate_danger:].reshape(-1,1), values[-1*self.values_to_calculate_danger:])
            slope = reg.coef_[0]

        if self.danger_mode == "ransac":
            ransac = RANSACRegressor(residual_threshold = self.residual_threshold).fit(times[-1*self.values_to_calculate_danger:].reshape(-1,1), values[-1*self.values_to_calculate_danger:])
            y_pred = ransac.predict(np.array([0,1])[:, np.newaxis])

            slope = (y_pred[1]-y_pred[0])/(1-0)
        return slope

    def get_danger_threshold(self):
        """
        Method to get danger threshold.
        """

        if not self.videos_to_get_danger_threshold is None:
            all_video_danger_data = []
            for video_name in self.videos_to_get_danger_threshold:
                video_danger_data = np.load(os.path.join(self.main_output_folder, video_name, "danger_values.npy"))
                all_video_danger_data.append(video_danger_data)

            print(all_video_danger_data)
            return np.percentile(np.array(all_video_danger_data), self.percentile_to_get_danger_threshold)
        else:
            return self.explicit_threshold

    def turn_to_line(self, values:np.ndarray):
        """
        Function to turn values into a line-like structure.
        values:np.ndarray -> Values to change.
        """
        return 1/np.sqrt(values)

    def is_relevant(self, values:np.ndarray):
        """
        Method to define if a track is relevant.
        values:np.ndarray -> Values to analyze.
        """
        return np.any(values>=self.relevance_threshold)

    def save_information(self, track_id:int, bbox:List[int], class_:int, confidence:float, time:int, image:np.ndarray) -> None:
        """
        Method to obtain the area and bbox center position from an object detected by an object detector.
        track_id:int -> object tracker identification.
        bbox:List[int] -> bbox assumed as [x,y,width,height] with corners (left-right, top-bottom) [x,y], [x+width,y], [x,y+height] and [x+width,y+height].
        class_:int -> Class value.
        confidence:float -> Value in [0,1] with the class_ confidence.
        time:int -> Time value of the image containing the object.
        image:np.ndarray -> RGB image as numpy matrix.
        """

        area = get_area_from_bbox(bbox)
        center = get_center_from_bbox(bbox)
        info = [bbox[0], bbox[1], bbox[2], bbox[3], center[0], center[1], area/(image.shape[0]*image.shape[1]), time, class_, confidence, -1, -1]

        get_danger = True
        if not track_id in self.information.keys():
            self.information[track_id] = []
        track_id_register = self.information[track_id]
        track_id_register.append(info)

        if len(track_id_register) >= self.track_length_to_calculate_danger:
            track_id_register_np = np.array(track_id_register)
            danger_value = self.calculate_danger(track_id_register_np[:,6], track_id_register_np[:,7])
            track_id_register[-1][-2] = danger_value
            dangerous = abs(danger_value) > abs(self.danger_threshold)
            print(dangerous)
            track_id_register[-1][-1] = dangerous
            self.all_danger_values.append(danger_value)
        self.information[track_id] = track_id_register

        return None
    
    def get_all_danger_values(self):
        return np.array(self.all_danger_values)

    def plot_aux_information(self, track_id:int, information:List, output:str) -> None:
        """
        Method to plot register information.
        track_id:int -> object tracker identification.
        information:List -> Generic list with the information to be plotted.
        output:str -> Folder to save plots in.
        relevant_only:bool -> Do we plot all tracks with enough information or do we filter them by relevant using self.relevant()? Default False.
        """


        information_np = np.array(information)
        sequence_times = information_np[:,7]
        dangerous = information_np[self.track_length_to_calculate_danger:,11] == 1
        if not self.relevant_only or self.is_relevant(information_np[:,6]):

            plt.clf()
            fig, ax1 = plt.subplots()
            ax1.scatter(sequence_times, information_np[:,6], color="yellowgreen")
            ax1.set_ylabel("Area")
            ax1.set_xlabel("Time")
            plt.savefig(f"{output}/{track_id}_orig.jpg",bbox_inches='tight')


            line_information = self.turn_to_line(information_np[:,6])
            plt.clf()
            #print(track_id)
            #print(np.array(information)[:,8])
            #print(np.array(information)[:,7])
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()

            if self.danger_mode == "ransac":

                ransac = RANSACRegressor(residual_threshold = self.residual_threshold).fit(sequence_times.reshape(-1,1), line_information)
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)

                ax1.scatter(sequence_times[inlier_mask], line_information[inlier_mask], color="yellowgreen", label= "Inliers")
                ax1.scatter(sequence_times[outlier_mask], line_information[outlier_mask], color="gold", label="Outliers")
                
                ax1.plot(sequence_times, ransac.predict(sequence_times.reshape(-1,1)), color='g', linewidth=2)

            if self.danger_mode == "linear_regression":
                ax1.scatter(sequence_times, line_information)

                reg = LinearRegression().fit(sequence_times.reshape(-1,1), line_information)
                
                ax1.plot(sequence_times, reg.predict(sequence_times.reshape(-1,1)), color='g', linewidth=2)

            ax1.set_ylabel(r"$\frac{1}{\delta}$", rotation=0, fontsize=20)
            ax1.set_xlabel("Time")

            danger_calculation_time_zone_sequence_times = sequence_times[self.track_length_to_calculate_danger:]
            danger_calculation_time_zone_danger = information_np[self.track_length_to_calculate_danger:,10]
            #ax2.plot(sequence_times, sequence_times.shape[0]*[self.danger_threshold], c='r')
            ax2.plot(sequence_times, sequence_times.shape[0]*[-1*self.danger_threshold], c='r')
            ax2.scatter(danger_calculation_time_zone_sequence_times[dangerous], danger_calculation_time_zone_danger[dangerous], c='r', marker='x', label="Dangerous")
            ax2.scatter(danger_calculation_time_zone_sequence_times[np.logical_not(dangerous)], danger_calculation_time_zone_danger[np.logical_not(dangerous)], c='r', marker='o', label="Non Dangerous")
            ax2.set_ylabel("Estimated Relative Speed")
            ax2.legend()
            plt.savefig(f"{output}/{track_id}_with_danger.jpg",bbox_inches='tight')

            plt.clf()
            fig, ax1 = plt.subplots()
            if self.danger_mode == "ransac":

                ransac = RANSACRegressor(residual_threshold = self.residual_threshold).fit(sequence_times.reshape(-1,1), line_information)
                inlier_mask = ransac.inlier_mask_
                outlier_mask = np.logical_not(inlier_mask)

                ax1.scatter(sequence_times[inlier_mask], line_information[inlier_mask], color="yellowgreen", label= "Inliers")
                ax1.scatter(sequence_times[outlier_mask], line_information[outlier_mask], color="gold", label="Outliers")
                
                ax1.plot(sequence_times, ransac.predict(sequence_times.reshape(-1,1)), color='g', linewidth=2)

            if self.danger_mode == "linear_regression":
                ax1.scatter(sequence_times, line_information)

                reg = LinearRegression().fit(sequence_times.reshape(-1,1), line_information)
                
                ax1.plot(sequence_times, reg.predict(sequence_times.reshape(-1,1)), color='g', linewidth=2)

            ax1.set_ylabel(r"$\frac{1}{\delta}$", rotation=0, fontsize=20)
            ax1.set_xlabel("Time")

            ax1.legend()
            plt.savefig(f"{output}/{track_id}.jpg",bbox_inches='tight')


    def plot_history_on_image(self, track_id:int, rgb_frame:np.ndarray, output_folder:str):
        """
        Method to plot historic register information for a track id.
        track_id:int -> object tracker identification.
        rgb_frame:np.ndarray -> RGB image as numpy matrix.
        output:str -> Folder to save plots in.
        """
        random
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        drawn_image = rgb_frame[:,:,[2,1,0]].copy()
        history_information = self.information[track_id]
        count = 0
        len_history = len(history_information)
        for info_index, info in enumerate(history_information):
            [x, y, width, height, _, _, area, time, class_, confidence, danger, dangerous] = info
            color = self.colors[track_id]
            if count == 0 or info_index==len_history-1:
                drawn_image = cv2.rectangle(drawn_image, (x,y), (x+width, y+height), (int(color[0]),int(color[1]),int(color[2])), 2)
            count+=1
            if count == self.number_of_skipped_prints_when_plotting_history_on_images:
                count = 0    
        cv2.imwrite(f"{output_folder}/{track_id}_{random.randint(0,10000)}.jpg", drawn_image)

    def analyze_information_sequence(information_sequence:List[Tuple[int,int,int]], fps:int, prediction_time_range:int = 3, output_graph_name:str = None) -> None:
        """
        Function to analyze a sequence of tuples. Each tuple structure is (x,y,area) with (x,y) as the position of the vehicle and area the bbox area.
        We will analyze each tuple as a position in a 3D space related to a time.
        """

        """
        sequence_length = len(information_sequence)
        sequence_times = np.arange(0, sequence_length/fps, 1/fps)
        reg = LinearRegression().fit(sequence_times, np.array(information_sequence))

        prediction_time = np.arange(sequence_times[-1], sequence_times[-1]+prediction_time_range, 1/fps)
        pred = reg.predict(prediction_time)
        if not output_graph_name is None:
            fig = plot_tuple_sequence(information_sequence, pred, f"{os.path.basename(output_graph_name)}")
            plt.set_current_figure(fig)
            plt.savefig(output_graph_name)
        """
        return None

    def analyze_information_dict(information_dict:Dict[int,List[Tuple[int,int,int]]], frame_index:int, fps:int, prediction_time_range:int = 3, output_analysis_folder:str = None)-> None:
        """
        for key in information_dict.keys():
            information_register = information_dict[key]
            analyze_information_tuple_sequence(information_register, fps, prediction_time_range, os.path.join(output_analysis_folder,f"frame_{frame_index}_track_{key}.jpg"))

        """
        return None