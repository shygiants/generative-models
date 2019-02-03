""" Trainer """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import debug as tfdbg
from tflibs.runner import Runner, DatasetInitializer, TrainInitializer
from tflibs.training import EvaluationRunHook
from tflibs.datasets import build_input_fn
from tflibs.utils import strip_dict_arg


def run(job_dir,
        train_iters,
        estimator,
        model_cls,
        dataset,
        train_batch_size,
        eval_batch_size,
        eval_steps,
        num_parallel_batches,
        shuffle_buffer_size,
        prefetch_buffer_size,
        no_eval,
        debug,
        debug_address):
    ############
    # Datasets #
    ############
    dataset_train = dataset.read(split='train')

    #######
    # Run #
    #######
    try:
        global_step = estimator.get_variable_value('global_step')
    except ValueError:
        global_step = 1

    tf.logging.info('Start training for %d.', global_step)

    hooks = []

    if not no_eval:
        dataset_test = dataset.read(split='test')
        hooks.append(EvaluationRunHook(estimator,
                                       build_input_fn(dataset_test,
                                                      eval_batch_size,
                                                      map_fn=strip_dict_arg(model_cls.eval_map_fn),
                                                      shuffle_and_repeat=False),
                                       eval_steps,
                                       summary=False))
    if debug:
        hooks.append(tfdbg.TensorBoardDebugHook(debug_address))

    # Run training for `train_iters` times
    estimator.train(build_input_fn(dataset_train,
                                   train_batch_size,
                                   map_fn=strip_dict_arg(model_cls.map_fn),
                                   num_parallel_batches=num_parallel_batches,
                                   shuffle_buffer_size=shuffle_buffer_size,
                                   prefetch_buffer_size=prefetch_buffer_size,
                                   global_step=global_step,
                                   shuffle_and_repeat=True),
                    max_steps=train_iters,
                    # Run evaluation every `eval_steps` iterations
                    hooks=hooks)


if __name__ == '__main__':
    runner = Runner(initializers=[
        DatasetInitializer(),
        TrainInitializer(),
    ])
    parser = runner.argparser

    ##############
    # Run Config #
    ##############
    parser.add_argument('--eval-steps',
                        type=int,
                        default=5000,
                        help='The number of steps to evaluate the model.')
    parser.add_argument('--num-parallel-batches',
                        type=int,
                        default=2,
                        help='The number of batches to process parallel.')
    parser.add_argument('--shuffle-buffer-size',
                        type=int,
                        default=10,
                        help='The size of buffer for shuffling.')
    parser.add_argument('--prefetch-buffer-size',
                        type=int,
                        default=10,
                        help='The size of buffer to prefetch.')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Whether to debug.')
    parser.add_argument('--debug-address',
                        default='grpc://localhost:6064',
                        help='The address of debugger server.')
    parser.add_argument('--no-eval',
                        action='store_true',
                        help='Whether to eval.')

    runner.run(run)
