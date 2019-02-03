"""
    MNIST
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tflibs.datasets import BaseDataset, ImageSpec, LabelSpec


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


export = MNIST
