""" Main runner """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tflibs.runner import Runner, DatasetInitializer, EvalInitializer
from tflibs.utils import strip_dict_arg
from tflibs.datasets import build_input_fn


def run(job_dir,
        estimator: tf.estimator.Estimator,
        model_cls,
        dataset,
        step,
        eval_index,
        eval_batch_size):
    ############
    # Datasets #
    ############
    dataset_test = dataset.read(split='test')

    #######
    # Run #
    #######
    checkpoints = tf.train.get_checkpoint_state(job_dir).all_model_checkpoint_paths

    def index_filter(fn, col):
        return list(map(lambda t: (lambda _, el: el)(*t), filter(lambda t: fn(*t), enumerate(col))))

    if step is not None:
        checkpoints = filter(lambda p: os.path.basename(p) == 'model.ckpt-{}'.format(step), checkpoints)
    elif eval_index is not None:
        checkpoints = index_filter(lambda i, _: i % 8 == eval_index, checkpoints)

    for checkpoint_path in sorted(checkpoints, reverse=True):
        step = os.path.basename(checkpoint_path).split('-')[-1]

        # Run evaluation
        tf.logging.info('Start evaluation for {}.'.format(step))
        estimator.evaluate(
            build_input_fn(dataset_test, eval_batch_size, map_fn=strip_dict_arg(model_cls.eval_map_fn),
                           shuffle_and_repeat=False),
            hooks=[],
            checkpoint_path=checkpoint_path)


if __name__ == '__main__':
    runner = Runner(initializers=[
        DatasetInitializer(),
        EvalInitializer(),
    ])
    parser = runner.argparser

    #########
    # Model #
    #########
    parser.add_argument('--step',
                        type=int,
                        default=None,
                        help='Step to save.')

    ##############
    # Run Config #
    ##############
    parser.add_argument('--eval-index',
                        type=int,
                        default=None,
                        help='Eval index.')

    runner.run(run)
