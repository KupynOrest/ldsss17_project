import os
import sys
import tensorflow as tf
from Obj_detector import Obj_detector


if __name__ == '__main__':
    directory_to_save = os.path.join(os.getcwd(), '..' ,'data', 'object_detections')
    if not os.path.exists(directory_to_save):
        os.makedirs(directory_to_save)
    directory_of_images = os.path.join(os.getcwd(), '..','data/data_subset/images')
    for category in os.listdir(directory_of_images):
        for frame in os.listdir(os.path.join(directory_of_images, category)):
            path_to_save = os.path.join(directory_to_save, category, frame)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            detector = Obj_detector(path_to_save, os.path.join(directory_of_images, category, frame))
            detector.fit_transform(os.listdir(os.path.join(directory_of_images, category, frame)))

