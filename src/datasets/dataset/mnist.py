"""
    MNIST
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tflibs.datasets import BaseDataset, ImageSpec, LabelSpec
from tflibs.ops import normalize


class MNIST(BaseDataset):
    def __init__(self, dataset_dir):
        BaseDataset.__init__(self, os.path.join(dataset_dir, 'mnist'))

    @property
    def tfrecord_filename(self):
        return 'mnist.tfrecord'

    def _init_feature_specs(self):
        return {
            'image': ImageSpec([28, 28, 1]),
            'label': LabelSpec(10),
        }

    @staticmethod
    def map_fn(image, label, _id):
        return {
                   'image': tf.image.convert_image_dtype(image, tf.float32),
                   '_id': _id,
               }, tf.to_float(label)

    @staticmethod
    def eval_map_fn(*args, **kwargs):
        return MNIST.map_fn(*args, **kwargs)


export = MNIST
