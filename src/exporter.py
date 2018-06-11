""" Exporter """

import os

import tensorflow as tf

from tflibs.ops import normalize
from tflibs.runner import Runner, ModelInitializer


def run(job_dir,
        model_cls,
        step):
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
        model_dir=job_dir)
    tf.logging.info(estimator)

    #######
    # Run #
    #######
    try:
        global_step = estimator.get_variable_value('global_step')
    except ValueError:
        global_step = 1

    tf.logging.info('Export a model for %d.', step or global_step)
    estimator.export_savedmodel(job_dir, serving_input_receiver_fn,
                                checkpoint_path=os.path.join(job_dir, 'model.ckpt-{}'.format(step)) if step else None)


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
