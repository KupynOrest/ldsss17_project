import glob
import os.path
import os
import logging
import itertools
import numpy as np


import torch
import torch.nn
import torchvision
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


def get_movies(path='data_subset', sub_dir='images'):
    movies = glob.glob(path + '/' + sub_dir + '/**', recursive=True)
    movies = [m for m in movies if m.endswith('.jpg')]
    movies = [(get_label(title), list(frames)) for title, frames in itertools.groupby(movies, get_title)]
    return movies


def prepare_model():
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = DummyLayer()
    model.train(False)
    if torch.cuda.is_available():
        return model.cuda()

    return model


def variable(t: torch.FloatTensor):
    if torch.cuda.is_available():
        return Variable(t.cuda())
    return Variable(t)


def store_features(in_dir, out_dir, frames_count=25):
    for i, (features, label, title) in \
            enumerate(get_class_features(in_dir=in_dir, frames_count=frames_count)):
        os.makedirs(os.path.join(out_dir, label), exist_ok=True)
        np.save(os.path.join(out_dir, label, title + '.npy'), features.numpy())


def get_class_features(in_dir, frames_count=25):
    model = prepare_model()
    movies = get_movies(in_dir)

    for label, frames in movies:
        title = os.path.basename(get_title(frames[0]))
        logger.info('Loading movie with category %s name %s and %d frames', label, title, len(frames))

        inputs = []
        frames = [frames[i] for i in range(0, len(frames), len(frames) // frames_count + 1)]

        for frame in frames:
            img = Image.open(frame)
            inputs.append(data_transforms(img).unsqueeze(0))

        if not inputs:
            continue

        features = model(variable(torch.cat(inputs))).data
        yield features, label, title


def get_class_features_for_batches(in_dir, filter_size=16, stride=8):
    model = prepare_model()
    movies = get_class_movies(in_dir)

    for label, frames in movies:
        title = get_title(frames[0]).split("/").pop()
        logger.info('Loading movie with category %s name %s and %d frames', label, title, len(frames))

        for i in range(0, len(frames), stride):
            if i + 16 < len(frames):
                output = []
                inputs = []
                title = get_title(frames[0]).split("/").pop() + '_' + str(i)

                frames_subset = frames[i: i+filter_size]
                for frame in frames_subset:
                    img = Image.open(frame)
                    inputs.append(data_transforms(img).unsqueeze(0))

                if not inputs:
                    continue

                features = model(variable(torch.cat(inputs))).data
                output.append(features.unsqueeze(0))

                yield torch.cat(output), title


if __name__ == '__main__':
    store_features('../data_subset', '../data/test_np_subset')
