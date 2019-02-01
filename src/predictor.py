""" Exporter """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import numpy as np

from tflibs.ops import normalize
from tflibs.runner import Runner, ModelInitializer
from tflibs.image import encode


def run(job_dir,
        model_cls,
        model_args,
        step,
        **kwargs):
    def serving_input_receiver_fn():
        decoded_image = tf.placeholder(dtype=tf.uint8,
                                       shape=[28, 28, 1],
                                       name='input_image')
        image = normalize(decoded_image)
        image = tf.expand_dims(image, axis=0)

        receiver_tensors = {'image': decoded_image}

        return tf.estimator.export.ServingInputReceiver({'image': image},
                                                        receiver_tensors)

    ##########
    # Models #
    ##########
    estimator = tf.estimator.Estimator(
        model_cls.model_fn,
        model_dir=job_dir,
        params={'model_args': model_args})
    tf.logging.info(estimator)

    #######
    # Run #
    #######
    out = estimator.predict(lambda: {'image': tf.zeros((1, 28, 28, 1))}, predict_keys='sample', yield_single_examples=False)
    for i, o in enumerate(out):
        if i == 10:
            return
        a = o['sample']
        tf.logging.info(a)
        with open(os.path.join(job_dir, 'sample{}.jpg'.format(i)), 'wb') as f:
            f.write(encode(np.concatenate([a]*3, axis=-1)))


if __name__ == '__main__':
    runner = Runner(initializers=[
        ModelInitializer(),
    ])
    parser = runner.argparser

    #########
    # Model #
    #########
    parser.add_argument('--step',
                        type=int,
                        default=None,
                        help='Step to save.')

    runner.run(run)
