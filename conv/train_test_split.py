import os
import shutil


def move_test_data(file_names, image_dir):
    """
    :type file_names: list[str]
    :return:
    """
    for file_name in file_names:
        file_name = file_name.strip().rstrip('.avi')
        print(file_name)

        shutil.move(os.path.join(image_dir, file_name), os.path.join('data/test', file_name))


def split_train_test():
    test_source = 'conv/data_subset/ucfTrainTestlist'
    image_dir = 'data/UCF101/images'

    with open(os.path.join(test_source, 'testlist01.txt')) as flist:
        file_names = flist.readlines()

    move_test_data(file_names, image_dir)
    shutil.move(image_dir, 'data/train')

split_train_test()