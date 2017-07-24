import os
import sys
sys.path.append('../')
from features.extractor import get_class_features


def convert_to_images():
    videos_folder = 'processing/videos/'
    images_folder = 'processing/images/'

    if not os.path.exists(images_folder):
        os.mkdir(images_folder)

    list_video = sorted(os.listdir(videos_folder))
    videos_list = []

    for v in list_video:
        path_class_folder_v_video = os.path.join(videos_folder, v)
        path_class_folder_im_video = os.path.join(images_folder, v)[:-4]  # remove extension from video file name
        if not os.path.exists(path_class_folder_im_video):
            os.mkdir(path_class_folder_im_video)
        image_name = path_class_folder_im_video + "/im_%03d.jpg"
        os.system("ffmpeg -i " + path_class_folder_v_video +
                  " -f image2 " + image_name)
        videos_list.append(path_class_folder_im_video)
    return videos_list


def run_process():
    video_images_list = convert_to_images()
    for path in video_images_list:
        features = get_class_features(path, frames_count=50)
        print(features.shape)

    return 1


run_process()