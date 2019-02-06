from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from tflibs.model import Model
from tflibs.training import Optimizer, Dispatcher, Optimizers
from tflibs.utils import param_consumer
from tflibs.ops import concat_images

from networks.gan import Generator, Discriminator


class GAN(Model):
    def __init__(self, features, model_idx=0, device=None, **hparams):
        Model.__init__(self, features, model_idx=model_idx, device=device, **hparams)

        self.networks.update(generator=Generator(**hparams),
                             discriminator=Discriminator(**hparams))

    @Model.image(summary='Real_Images')
    def real_images(self):
        return self.features['image']

    @Model.image(summary='Fake_Images')
    def fake_images(self):
        return self.networks.generator(self.noise)

    @Model.image
    def sampled_images(self):
        return tf.reshape(self.networks.generator(self.features['z']), (28, 28, 1))

    @Model.tensor
    def noise(self):
        return tf.distributions.Uniform(-3 ** 0.5, 3 ** 0.5).sample((tf.shape(self.real_images)[0], self.hparams.dim_z))

    @Model.tensor(summary='Logits_Real')
    def logits_real(self):
        return self.networks.discriminator(self.real_images)

    @Model.tensor(summary='Logits_Fake')
    def logits_fake(self):
        return self.networks.discriminator(self.fake_images)

    @Model.loss
    def g_loss(self):
        return tf.losses.log_loss(tf.ones_like(self.logits_fake), self.logits_fake, scope='G_Loss')

    @Model.loss(summary='D_Loss')
    def d_loss(self):
        d_loss_real = tf.losses.log_loss(tf.ones_like(self.logits_real), self.logits_real, weights=0.5,
                                         scope='D_Loss_Real')
        d_loss_fake = tf.losses.log_loss(tf.zeros_like(self.logits_fake), self.logits_fake, weights=0.5,
                                         scope='D_Loss_Fake')

        return d_loss_real + d_loss_fake

    @Model.loss
    def loss(self):
        return self.g_loss + self.d_loss

    @classmethod
    def train(cls, features: dict, learning_rate, **hparams):
        def decay_fn(gs, lr):
            return lr * (1 / 1.000004 ** gs)

        d_optimizer = Optimizer(learning_rate, 'Discriminator', decay_policy='lambda',
                                decay_params={'lr_fn': decay_fn})
        g_optimizer = Optimizer(learning_rate, 'Generator', decay_policy='lambda',
                                decay_params={'lr_fn': decay_fn})

        dispatcher = Dispatcher(cls, hparams, features, num_towers=1, model_parallelism=False)

        d_train_op = dispatcher.minimize(d_optimizer, lambda m: m.d_loss)
        g_train_op = dispatcher.minimize(g_optimizer, lambda m: m.g_loss, depends=d_train_op)

        # Increment global step
        global_step = tf.train.get_or_create_global_step()
        with tf.control_dependencies([g_train_op]):
            step_incr = tf.assign_add(global_step, 1, name='step_incr')

        train_op = step_incr

        chief = dispatcher.chief  # typd: GAN
        tf.logging.info('Explicitly declared summaries')

        loss = chief.loss
        chief.summary_loss()

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

    @classmethod
    def evaluate(cls, features: dict, gpu_eval=None, **hparams):
        chief = cls(features, device=gpu_eval if gpu_eval is not None else 0, **hparams)

        # TODO: Metrics
        metrics = {}

        return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL,
                                          loss=chief.loss,
                                          eval_metric_ops=metrics)

    @classmethod
    def predict(cls, features: dict, gpu_eval=None, **hparams):
        chief = cls(features, device=gpu_eval if gpu_eval is not None else 0, **hparams)

        predictions = {
            'sample': chief.sampled_images
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

        add_arg_scope(tf.layers.dropout)

        with tf.variable_scope('GAN', values=[features]):
            with arg_scope([tf.layers.dropout], training=mode == tf.estimator.ModeKeys.TRAIN):
                if mode == tf.estimator.ModeKeys.TRAIN:
                    train_args = params['train_args']
                    batch_size = train_args['train_batch_size']
                    model_args.update(batch_size=batch_size, **train_args)
                    return GAN.train(features, **model_args)
                elif mode == tf.estimator.ModeKeys.EVAL:
                    eval_args = params['eval_args']
                    batch_size = eval_args['eval_batch_size']
                    model_args.update(batch_size=batch_size, **eval_args)
                    return GAN.evaluate(features, **model_args)
                elif mode == tf.estimator.ModeKeys.PREDICT:
                    return GAN.predict(features, **model_args)
                else:
                    raise ValueError

    @staticmethod
    def map_fn(image, label, _id):
        return {
                   'image': tf.image.convert_image_dtype(image, tf.float32),
                   '_id': _id,
               }, tf.to_float(label)

    @staticmethod
    def eval_map_fn(*args, **kwargs):
        return GAN.map_fn(*args, **kwargs)

    @classmethod
    def add_model_args(cls, argparser, parse_args):
        argparser.add_argument('--dim-z',
                               type=int,
                               default=100,
                               help='The dimension of latent `z`')
        argparser.add_argument('--num-filters',
                               type=int,
                               default=64,
                               help='The number of hidden units')

    @classmethod
    def add_train_args(cls, argparser, parse_args):
        #############
        # Optimizer #
        #############
        argparser.add_argument('--momentum',
                               type=float,
                               default=0.5,
                               help='Momentum.')
        ############
        # Training #
        ############
        argparser.add_argument('--learning-rate',
                               type=float,
                               default=0.1,
                               help='Learning rate.')

    @classmethod
    def add_eval_args(cls, argparser, parse_args):
        argparser.add_argument('--gpu-eval',
                               type=str,
                               help='GPU ids for evaluation.',
                               default=None)


export = GAN
