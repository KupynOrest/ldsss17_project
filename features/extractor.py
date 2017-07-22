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



def get_features():
    model = prepare_model()

    for filename in get_movies('../data_subset/videos'):
        logger.info('Loading file %s', filename)

        category = os.path.dirname(filename)
        matrix = []
        for frame in get_frames(filename):
            if frame is None:
                continue
            img = Image.fromarray(frame)
            inputs = data_transforms(img)
            outputs = model(Variable(inputs.resize_(1, 3, 224, 224)))
            #print(outputs.data.view(512).numpy())
            matrix.append(outputs.data.view(512).numpy())

        if not matrix:
            continue

        arr = np.ndarray(matrix)
        logger.info('For file %s processed features %s', filename, arr.size)

        path = 'data_features/{}'.format(category)
        os.makedirs(path)
        np.save('{}/{}.npy'.format(path, os.path.basename(filename)), arr)

if __name__ == '__main__':
    get_features()
