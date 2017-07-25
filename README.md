# Video Action Recognition using ConvNet + LSTM

This repository contains the code for the video action recognition trained on [UCF-101](http://crcv.ucf.edu/data/UCF101.php) dataset.

## Preprocessing

We split the video into the chunks of 16 frames with the stride of 8. We train the model on all of the chunks of the video and in test mode average the predictions from all chunks for 1 video. This is a form of a temporal data augmentation for video classification and helps to generalize better.

## Architecture

We've implemented ConvNet + LSTM model for action recognition like this ![alt text](https://raw.githubusercontent.com/lyubonko/ldsss17_project/master/images/architecture.png) 

We use [ResNet-18](http://pytorch.org/docs/master/torchvision/models.html) pretrained model from torchvision as a feature extractor. The last fully connected layer for classification is removed and we use 512 featured for each frame taken from the activation of the last ResNet layer. The features for 16 frames are then fed to many-to-one [Batch-Norm LSTM](https://arxiv.org/pdf/1603.09025.pdf) which performs the final classification. 

## Dependencies

* Python 3+
* CPU or NVIDIA GPU + CUDA CuDNN
* Torch
