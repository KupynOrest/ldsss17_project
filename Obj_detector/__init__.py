import os
import time
from Obj_detector import Obj_detector


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    directory_to_save = os.path.join(os.getcwd(), '..' ,'data', 'object_detections')
    if not os.path.exists(directory_to_save):
        os.makedirs(directory_to_save)
    directory_of_images = os.path.join(os.getcwd(), '..','data/data_subset/images')
    start = time.time()
    for category in os.listdir(directory_of_images):
        start2 = time.time()
        for frame in os.listdir(os.path.join(directory_of_images, category)):
            start3 = time.time()
            path_to_save = os.path.join(directory_to_save, category, frame)
            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
            detector = Obj_detector(path_to_save, os.path.join(directory_of_images, category, frame))
            detector.fit_transform(os.listdir(os.path.join(directory_of_images, category, frame)))
            print('{} ---------- proceded ---------- {}'.format(frame, time.time() - start3))
        print('{} -------------------- PROCESED -------------------- {}'.format(category, time.time() - start2))
    print('TOTAL TIME ----------------------------- {} --------------------'.format(time.time() - start))
