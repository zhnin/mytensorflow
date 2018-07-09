# @Time    : 2018/7/8 10:23
# @Author  : cap
# @FileName: retrain.py
# @Software: PyCharm Community Edition
# @introduction:
import argparse
import os
import re
import sys
import tensorflow as tf
from tensorflow.python.platform import gfile


def prepare_file_syatem():
    pass


def create_model_info(architecture):
    pass


def maybe_download_and_extract(data_url):
    """download and extract model tar file"""
    pass


def create_model_graph(model_info):
    """create a graph from saved graphdef file and returns a graph object"""
    return None, None, None


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """builds a list of training iages form the file system"""
    if not gfile.Exists(image_dir):
        tf.logging.error('Image directory' + image_dir + 'not found.')
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        tf.logging.info('Looking for images in ' + dir_name + '')
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            tf.logging.warning('No files found')
            continue
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
def main(_):
    # logging
    tf.logging.set_verbosity(tf.logging.INFO)

    # prepare necessary directories that can be used
    prepare_file_syatem()

    # gather information about the model architecture we'll use
    model_info = create_model_info(FLAGS.architecture)

    # set up the pre-trained graph
    maybe_download_and_extract(model_info['data_url'])
    graph, bottleneck_tensor, resized_image_tensor = (create_model_graph(model_info))

    # look at the folder structure, and create lists of all the images.
    image_list = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main, argv=[sys.argv[0]] + unparsed)