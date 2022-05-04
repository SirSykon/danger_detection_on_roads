"""
Author: Jorge García <jrggcgz@gmail.com>


"""
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import numpy as np
from object_detection.print_utils import print_utils

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
    colors:Dict[int,List[int] -> Current color dictionary.
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
    center = [x+width/2, y+height/2]
    return center

def plot_area_information(areas:List[int], color) -> plt.figure:
    sequence_length = len(areas)
    numpy_sequence = np.array(areas)
    fig, ax1 = plt.subplots(1,1)
    ax1.plot(range(sequence_length), numpy_sequence, c=color)
    ax2.plot(numpy_sequence[:,1], numpy_sequence[:,2], c=color)

    return fig

def plot_tuple_sequence(tuple_sequence:List[Tuple[int,int,int]], prediction_sequence:np.ndarray, plot_name:str, color:List[int]=[128.,128.,128.]) -> None:
    sequence_length = len(tuple_sequence)
    numpy_sequence = np.array(tuple_sequence)
    prediction_length = prediction_sequence.shape[0]
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.suptitle(plot_name)
    ax1.plot(range(sequence_length), numpy_sequence[:,0], c=color)
    ax2.plot(numpy_sequence[:,1], numpy_sequence[:,2], c=color)

    ax1.plot(range(sequence_length,sequence_length+prediction_length), prediction_sequence[:,0], c='r')
    ax2.plot(range(sequence_length,sequence_length+prediction_length), prediction_sequence[:,1], c='r')
    ax3.plot(range(sequence_length,sequence_length+prediction_length), prediction_sequence[:,2], c='r')

    return fig