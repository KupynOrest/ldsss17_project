import os
import numpy
from extractor import get_class_features

for directory in ['../data/train', '../data/test']:
    classes = sorted(os.listdir(directory))[1:]
    for class_name in classes:
        in_dir = os.path.join(directory, class_name)
        out_dir = os.path.join(directory + '_np_50n', class_name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            for i, (features, label, title) in enumerate(get_class_features(in_dir, frames_count=50, sub_dir='*')):
                np_data = features.cpu().numpy()
                numpy.save(os.path.join(out_dir, title + '.npy'), np_data)


