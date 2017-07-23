import os
import numpy
from features.extractor import get_class_features

sequence_length = 157

for directory in ['../data/train', '../data/test']:
    classes = sorted(os.listdir(directory))[1:]
    for class_name in classes:
        in_dir = os.path.join(directory, class_name)
        out_dir = os.path.join(directory + '_np', class_name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i, (labels, images, title) in enumerate(get_class_features(in_dir=in_dir)):
            np_data = images.numpy()

            numpy.save(os.path.join(out_dir, title + '.npy'), np_data)






