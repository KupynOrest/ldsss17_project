import os
import numpy
from extractor import get_class_features_for_batches

#frames_median = 160

for directory in ['../data/train', '../data/test']:
    classes = sorted(os.listdir(directory))[1:]
    for class_name in classes:
        in_dir = os.path.join(directory, class_name)
        out_dir = os.path.join(directory + '_np_aug', class_name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            for i, (images, title) in \
                    enumerate(get_class_features_for_batches(in_dir=in_dir)):
                np_data = images.cpu().numpy()

                numpy.save(os.path.join(out_dir, title + '.npy'), np_data)






