import cv2
import glob
import os.path
import os
import logging
import torch.nn
import torchvision
import numpy as np

from torchvision import transforms
from torch.autograd import Variable

from PIL import Image


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


data_transforms = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DummyLayer(torch.nn.Module):
    def forward(self, input_):
        return input_


def get_movies(path='data_subset/videos'):
    for filename in glob.iglob(path + '/**', recursive=True):
        yield filename


def get_frames(filename):
    v = cv2.VideoCapture(filename)
    if not v.isOpened():
        logger.warning('Unable to find file %s', filename)
        return

    while v.isOpened():
        _, frame = v.read()
        yield frame


def prepare_model():
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = DummyLayer()
    return model


def get_features(in_dir, out_dir):
    model = prepare_model()

    for filename in get_movies(in_dir):
        logger.info('Loading file %s', filename)

        category = os.path.basename(os.path.dirname(filename))
        frames = []
        for frame in get_frames(filename):
            if frame is None:
                break

            img = Image.fromarray(frame)
            frames.append(data_transforms(img).unsqueeze(0))

        if not frames:
            continue

        features = model(Variable(torch.cat(frames))).data

        logger.info('For file %s processed features %s', filename, features.size())

        path = '{}/{}'.format(out_dir, category)
        os.makedirs(path, exist_ok=True)
        np.save('{}/{}.npy'.format(path, os.path.basename(filename)), features.numpy())

if __name__ == '__main__':
    get_features('../data_subset/videos', '../data_features')
