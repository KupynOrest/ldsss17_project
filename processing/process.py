import os, sys
lib_path = os.path.abspath(os.path.join('features'))
sys.path.append(lib_path)
from extractor import get_class_features

def convert_to_images():
    videos_folder = 'processing/videos/'
    images_folder = 'processing/images/'

    if not os.path.exists(images_folder):
        os.mkdir(images_folder)

    list_video = sorted(os.listdir(videos_folder))
    video_images_list = []

    for v in list_video:
        path_class_folder_v_video = os.path.join(videos_folder, v)
        path_class_folder_im_video = os.path.join(images_folder, v)[:-4]  # remove extension from video file name
        video_images_list.append(path_class_folder_im_video)
        if os.path.exists(path_class_folder_im_video):
            print('skipping converting to images: ', v)
            continue

        os.mkdir(path_class_folder_im_video)
        image_name = path_class_folder_im_video + "/im_%03d.jpg"
        os.system("ffmpeg -i " + path_class_folder_v_video +
                  " -f image2 " + image_name)
    return video_images_list


def run_process():
    video_images_list = convert_to_images()

    for path in video_images_list:
        for i, (features, label, title) in enumerate(get_class_features(path, frames_count=50, sub_dir="")):
            print(features, label, title)

    return 1


run_process()