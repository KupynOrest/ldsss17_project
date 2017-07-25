import os, sys
lib_path = os.path.abspath(os.path.join('features'))
sys.path.append(lib_path)
from extractor import get_features_by_fps
import torch
import torch.nn as nn

from torch.autograd import Variable

opts = {
    'hidden_size': 512,
    'input_size': 512,
    'num_layers': 1,
    'dropout': 0.75,
    'batch_size': 128,
    'num_classes': 101,
    'sequence_length': 50
}

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


def run_on_model(inputs):
    use_gpu = torch.cuda.is_available()
    model = torch.load('processing/models/lstm68_11.pt')
    inputs_view = inputs.view(-1, opts.sequence_length, opts.input_size)
    if use_gpu:
        model = model.cuda()
        inputs_view = inputs_view.cuda()
    inputs = Variable(inputs_view)
    outputs = model(inputs)
    print(outputs)
    _, preds = torch.max(outputs.data, 1)
    print(_)
    print(preds)


def run_process():
    video_images_list = convert_to_images()

    for path in video_images_list:
        for i, (features, label, title) in enumerate(get_features_by_fps(path, frames_median=200, fps=6)):
            run_on_model(features)
    return


run_process()