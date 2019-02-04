""" Variational Auto-Encoder """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from tflibs.model import Network
from tflibs.nn import linear, conv2d, deconv2d, residual_block, Padding, Nonlinear, Norm, DeconvMethod


class Encoder(Network):
    def __init__(self, scope='Encoder', **hparams):
        Network.__init__(self, **hparams)
        self._scope = scope
        add_arg_scope(tf.layers.dense)

    @property
    def scope(self):
        return self._scope

    def __call__(self, inputs):
        # Hyperparameters
        dim_z = self.hparams.dim_z
        num_hidden = self.hparams.num_hidden

        with tf.variable_scope(self.scope, values=[inputs], reuse=tf.AUTO_REUSE):
            with arg_scope([tf.layers.dense],
                           kernel_initializer=tf.initializers.random_normal(stddev=0.1),
                           bias_initializer=tf.initializers.random_normal(stddev=0.1)):
                inputs = tf.layers.flatten(inputs)
                hidden = linear(inputs, num_hidden, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.Tanh)
                output = linear(hidden, dim_z * 2, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.NONE)

                mu, sigma = tf.split(output, 2, axis=-1)
                sigma = tf.sqrt(tf.math.exp(sigma))

                return tf.distributions.Normal(mu, sigma + 1e-6)


class Decoder(Network):
    def __init__(self, scope='Decoder', **hparams):
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
            with arg_scope([tf.layers.dense],
                           kernel_initializer=tf.initializers.random_normal(stddev=0.1),
                           bias_initializer=tf.initializers.random_normal(stddev=0.1)):
                hidden = linear(inputs, num_hidden, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.Tanh)
                output = linear(hidden, data_size, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.Sigmoid)

                return output
