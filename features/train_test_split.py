import os
import shutil


def move_test_data(file_names, image_dir, output_test):
    """
    :type file_names: list[str]
    :return:
    """
    for file_name in file_names:
        file_name = file_name.strip()
        shutil.move(os.path.join(image_dir, file_name), os.path.join(output_test, file_name))


def split_train_test(test_source='data/UCF101/ucfTrainTestlist', image_dir='data/UCF101/images',
                     output_train='data/train', output_test='data/test'):
    with open(os.path.join(test_source, 'testlist01.txt')) as flist:
        file_names = flist.readlines()

    move_test_data(file_names, image_dir, output_test)
    shutil.move(image_dir, output_train)
