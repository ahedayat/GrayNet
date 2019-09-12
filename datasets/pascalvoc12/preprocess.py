import os
import shutil
from optparse import OptionParser


def mkdir(dir_path, dir_name, forced_remove=False):
    new_dir = '{}/{}'.format(dir_path, dir_name)
    if forced_remove and os.path.isdir(new_dir):
        shutil.rmtree(new_dir)
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)


def touch(file_path, file_name, forced_remove=False):
    new_file = '{}/{}'.format(file_path, file_name)
    assert os.path.isdir(
        file_path), ' \"{}\" does not exist.'.format(file_path)
    if forced_remove and os.path.isfile(new_file):
        os.remove(new_file)
    if not os.path.isfile(new_file):
        open(new_file, 'a').close()


def write_file(file_path, file_name, content, forced_remove=True):
    touch(file_path, file_name, forced_remove=forced_remove)
    with open('{}/{}'.format(file_path, file_name), 'a') as f:
        f.write('{}\n'.format(content))


def copy_file(src_path, src_file_name, dst_path, dst_file_name):
    shutil.copyfile('{}/{}'.format(src_path, src_file_name),
                    '{}/{}'.format(dst_path, dst_file_name))


def ls(dir_path):
    return os.listdir(dir_path)


def save_data(data_mode, pascal_images_dir, pascal_masks_dir, saving_dst='.', saving_point=(0., 1.)):
    assert saving_point[0] <= saving_point[1] and saving_point[0] >= 0 and saving_point[1] <= 1, 'saving point error: 0<={}<={}<=1 ?!'.format(
        saving_point[0], saving_point[1])
    images_name = ls('{}'.format(pascal_images_dir))

    starting_point = saving_point[0] * len(images_name)
    ending_point = saving_point[1] * len(images_name)
    num_data = ending_point - starting_point

    mkdir('{}'.format(saving_dst),
          'images', forced_remove=False)
    touch(saving_dst, 'file_names.txt', forced_remove=True)

    for ix, (image_name) in enumerate(images_name):
        if ix < starting_point:
            continue
        if ix > ending_point:
            break

        copy_file(pascal_images_dir,
                  image_name,
                  '{}/images'.format(saving_dst),
                  image_name)
        write_file(saving_dst, 'file_names.txt',
                   image_name, forced_remove=False)

        print('%s: %d/%d( %.2f %% )' % (data_mode,
                                        ix,
                                        num_data,
                                        ix / num_data * 100
                                        ), end='\r')
    print()


def preprocess(data_dir, data_mode, saving_points, saving_dsts):

    assert data_mode in [
        'trainval', 'test'], 'Unknown data_mode. data_mode must be one of [\'trainval\', \'test\']'
    pascal_images_dir = '{}/JPEGImages'.format(data_dir)
    pascal_masks_dir = '{}/SegmentationClass'.format(data_dir)

    data_modes = ['train', 'val'] if data_mode == 'trainval' else ['test']
    assert len(data_modes) == len(
        saving_points), 'saving points must has {} saving point.'.format(len(data_mode))
    assert len(data_modes) == len(
        saving_dsts), 'saving destinations must has {} saving destination.'.format(len(data_mode))

    for (saving_dst, data_mode) in zip(saving_dsts, data_modes):
        mkdir(saving_dst, data_mode)

    for (saving_dst, data_mode, saving_point) in zip(saving_dsts, data_modes, saving_points):
        save_data(data_mode, pascal_images_dir, pascal_masks_dir,
                  saving_dst='{}/{}'.format(saving_dst, data_mode), saving_point=saving_point)


def _main(args):
    pascal_trainval_path = args.trainval
    pascal_test_path = args.test

    trainval_saving_points = [(0., 0.8), (0.8, 1.)]
    test_saving_points = [(0., 1.)]

    trainval_saving_dst = ['.', '.']
    test_saving_dst = '.'

    preprocess(pascal_trainval_path, 'trainval',
               trainval_saving_points, trainval_saving_dst)
    # preprocess(pascal_test_path, 'test', test_saving_points, test_saving_dst)


def get_args():
    parser = OptionParser()
    parser.add_option('--trainval-data', dest='trainval', default='./voc_trainval', type='string',
                      help='trainval data path')
    parser.add_option('--test-data', dest='test', default='./voc_test', type='string',
                      help='test data path')
    (options, args) = parser.parse_args()
    return options


if __name__ == "__main__":
    args = get_args()
    _main(args)
