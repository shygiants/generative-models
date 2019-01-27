from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tflibs.model import Model
from tflibs.training import Optimizer, Dispatcher
from tflibs.utils import param_consumer

from networks.deep_conv_net import DeepConvNet as DCN


class DeepConvNet(Model):
    def __init__(self, features, labels=None, model_idx=0, device=None, **hparams):
        Model.__init__(self, features, labels=labels, model_idx=model_idx, device=device, **hparams)

        # Define networks
        self.networks.update(dcn=DCN(scope='DeepConvNet', **hparams))

    @Model.image(summary='Input_Images')
    def images(self):
        return self.features['image']

    @Model.tensor(summary='Logits')
    def logits(self):
        return self.networks.dcn(self.images)

    @Model.tensor
    def predictions(self):
        return tf.argmax(self.scores, axis=1)

    @Model.tensor
    def scores(self):
        return tf.nn.softmax(self.logits)

    @Model.loss
    def loss(self):
        return tf.losses.softmax_cross_entropy(self.labels, self.logits, scope='Loss')

    @classmethod
    def num_devices(cls):
        return 1

    @classmethod
    def train(cls, features: dict, labels, learning_rate, **hparams):

        optimizer_params = param_consumer(['beta1', 'beta2'], hparams)
        decay_params = param_consumer(['train_iters', 'decay_iters', 'decay_steps'], hparams)

        optimizer = Optimizer(learning_rate, '', optimizer_params=optimizer_params, decay_params=decay_params)

        dispatcher = Dispatcher(cls, hparams, features, labels=labels, num_towers=1, model_parallelism=False)

        train_op = dispatcher.minimize(optimizer, lambda m: m.loss, global_step=tf.train.get_or_create_global_step())

        chief = dispatcher.chief  # type: DeepConvNet
        tf.logging.info('Explicitly declared summaries')
        loss = chief.loss
        chief.summary_loss()

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

    @classmethod
    def evaluate(cls, features: dict, labels, gpu_eval=None, **hparams):
        chief = cls(features, labels=labels, device=gpu_eval if gpu_eval is not None else 0, **hparams)

        metrics = {
            'accuracy': tf.metrics.accuracy(tf.argmax(labels, axis=1), chief.predictions)
        }

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL,
                                          loss=chief.loss,
                                          eval_metric_ops=metrics)

    @classmethod
    def predict(cls, features: dict, gpu_eval=None, **hparams):
        chief = cls(features, device=gpu_eval if gpu_eval is not None else 0, **hparams)

        predictions = {
            'score': chief.scores,
            'predictions': chief.predictions,
        }

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
                                          predictions=predictions,
                                          export_outputs={
                                              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                  tf.estimator.export.PredictOutput(predictions),
                                          })

    @staticmethod
    def model_fn(features, labels, mode, params):
        model_args = params['model_args'] if 'model_args' in params else {}

        if mode == tf.estimator.ModeKeys.TRAIN:
            model_args.update(params['train_args'])
            return DeepConvNet.train(features, labels, **model_args)
        elif mode == tf.estimator.ModeKeys.EVAL:
            model_args.update(params['eval_args'])
            return DeepConvNet.evaluate(features, labels, **model_args)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return DeepConvNet.predict(features, **model_args)
        else:
            raise ValueError

    @classmethod
    def add_train_args(cls, argparser, parse_args):
        #############
        # Optimizer #
        #############
        argparser.add_argument('--beta1',
                               type=float,
                               default=0.5,
                               help='Beta1 for Adam optimizer.')
        argparser.add_argument('--beta2',
                               type=float,
                               default=0.999,
                               help='Beta1 for Adam optimizer.')

        ############
        # LR Decay #
        ############
        argparser.add_argument('--decay-iters',
                               type=int,
                               default=100000,
                               help='The number of training iterations to decay learning rate.')
        argparser.add_argument('--decay-steps',
                               type=int,
                               default=1000,
                               help='The number of training steps to decay learning rate.')
        ############
        # Training #
        ############
        argparser.add_argument('--learning-rate',
                               type=float,
                               default=0.0001,
                               help='Learning rate.')

    @classmethod
    def add_eval_args(cls, argparser, parse_args):
        argparser.add_argument('--gpu-eval',
                               type=str,
                               help='GPU ids for evaluation.',
                               default=None)


export = DeepConvNet
