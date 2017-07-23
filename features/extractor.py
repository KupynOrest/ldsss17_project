import glob
import os.path
import os
import logging
import itertools
import random

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
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class DummyLayer(torch.nn.Module):
    def forward(self, input_):
        return input_


def get_label(title):
    return os.path.basename(os.path.dirname(title))


def get_title(filename):
    return os.path.dirname(filename)


def get_movies(path='data_subset', sub_dir='images', shuffle=True):
    movies = glob.glob(path + '/' + sub_dir + '/**', recursive=True)
    movies = [m for m in movies if m.endswith('.jpg')]
    movies = [(get_label(title), list(frames)) for title, frames in itertools.groupby(movies, get_title)]
    if shuffle:
        random.shuffle(movies)

    return movies


def get_frames_median(path, sub_dir='images'):
    return np.median([len(frames) for _, frames in get_movies(path=path, sub_dir=sub_dir, shuffle=False)])


def select_center(frames, median):
    if median >= len(frames):
        return frames

    padding = int((len(frames) - median) // 2)
    return frames[padding:padding + median]


def prepare_model():
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = DummyLayer()
    return model


def get_features(in_dir, batch_size):
    model = prepare_model()
    frames_median = int(get_frames_median(in_dir))
    movies = iter(get_movies(in_dir))

    while True:
        batch = itertools.islice(movies, 0, batch_size)
        output = []
        for label, frames in batch:
            logger.info('Loading movie with category %s and %d frames', label, len(frames))

            inputs = []
            for frame in select_center(frames, frames_median):
                img = Image.open(frame)
                inputs.append(data_transforms(img).unsqueeze(0))

            if not inputs:
                continue

            features = model(Variable(torch.cat(inputs))).data

            # left padding with zeros
            if len(frames) < frames_median:
                features = torch.cat([torch.zeros(frames_median - len(frames), 512), features])

            output.append(features.unsqueeze(0))

        return torch.cat(output)


if __name__ == '__main__':
    res = get_features('../data_subset', batch_size=5)
    print(res.size())
