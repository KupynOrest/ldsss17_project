import os
import numpy
from extractor import get_features_by_fps

for directory in ['../data/train', '../data/test']:
    classes = sorted(os.listdir(directory))[1:]
    for class_name in classes:
        in_dir = os.path.join(directory, class_name)
        out_dir = os.path.join(directory + '_np_70fps8', class_name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            for i, (features, label, title) in enumerate(get_features_by_fps(in_dir, frames_median=210, fps=8)):
                np_data = features.cpu().numpy()
                numpy.save(os.path.join(out_dir, title + '.npy'), np_data)


