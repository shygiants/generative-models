""" Generative Adversarial Networks """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from tflibs.model import Network
from tflibs.nn import linear, conv2d, deconv2d, residual_block, Padding, Nonlinear, Norm, DeconvMethod


class Discriminator(Network):
    def __init__(self, scope='Discriminator', **hparams):
        Network.__init__(self, **hparams)
        self._scope = scope
        add_arg_scope(tf.layers.dense)

    @property
    def scope(self):
        return self._scope

    def __call__(self, inputs):
        # Hyperparameters
        num_hidden = self.hparams.num_hidden

        with tf.variable_scope(self.scope, values=[inputs], reuse=tf.AUTO_REUSE):
            # TODO: Implement
            # TODO: Initialization
            with arg_scope([tf.layers.dense],
                           kernel_initializer=tf.initializers.random_uniform(minval=-.005, maxval=.005),
                           bias_initializer=tf.initializers.constant()):
                inputs = tf.layers.flatten(inputs)

                # TODO: Dropout
                inputs = linear(inputs, num_hidden, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.NONE)
                inputs = tf.expand_dims(inputs, axis=-1)
                inputs = tf.layers.max_pooling1d(inputs, 5, 5)
                inputs = tf.layers.flatten(inputs)
                inputs = tf.layers.dropout(inputs, 0.2)

                # TODO: Dropout
                inputs = linear(inputs, num_hidden, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.NONE)
                inputs = tf.expand_dims(inputs, axis=-1)
                inputs = tf.layers.max_pooling1d(inputs, 5, 5)
                inputs = tf.layers.flatten(inputs)
                inputs = tf.layers.dropout(inputs, 0.5)

                # TODO: Dropout
                inputs = linear(inputs, 1, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.Sigmoid)

                return inputs


class Generator(Network):
    def __init__(self, scope='Generator', **hparams):
        Network.__init__(self, **hparams)
        self._scope = scope
        add_arg_scope(tf.layers.dense)

    @property
    def scope(self):
        return self._scope

    def __call__(self, inputs):
        # Hyperparameters
        num_hidden = self.hparams.num_hidden
        data_size = self.hparams.data_size

        with tf.variable_scope(self.scope, values=[inputs], reuse=tf.AUTO_REUSE):
            # TODO: Implement
            with arg_scope([tf.layers.dense],
                           kernel_initializer=tf.initializers.random_uniform(minval=-.05, maxval=.05),
                           bias_initializer=tf.initializers.constant()):
                inputs = linear(inputs, num_hidden, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.ReLU)
                inputs = linear(inputs, num_hidden, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.ReLU)
                inputs = linear(inputs, data_size, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.Sigmoid)

                inputs = tf.reshape(inputs, (-1, 28, 28, 1))

                return inputs
