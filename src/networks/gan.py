""" Generative Adversarial Networks """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
from tensorflow.contrib.framework import arg_scope, add_arg_scope

from tflibs.model import Network
from tflibs.nn import linear, conv2d, deconv2d, residual_block, Padding, Nonlinear, Norm, DeconvMethod


class Discriminator(Network):
    def __init__(self, scope='Discriminator', **hparams):
        Network.__init__(self, **hparams)
        self._scope = scope
        add_arg_scope(tf.layers.conv2d)

    @property
    def scope(self):
        return self._scope

    def __call__(self, inputs):
        # Hyperparameters
        num_filters = self.hparams.num_filters

        with tf.variable_scope(self.scope, values=[inputs], reuse=tf.AUTO_REUSE):
            with arg_scope([tf.layers.conv2d],
                           kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                           bias_initializer=tf.initializers.constant()):
                leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)

                inputs = conv2d(inputs, num_filters, 3, strides=2, norm_fn=Norm.NONE, non_linear_fn=leaky_relu)
                inputs = conv2d(inputs, num_filters * 2, 3, strides=2, norm_fn=Norm.NONE, non_linear_fn=leaky_relu)

                inputs = tf.layers.flatten(inputs)

                inputs = linear(inputs, 1, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.Sigmoid)

                return inputs


class Generator(Network):
    def __init__(self, scope='Generator', **hparams):
        Network.__init__(self, **hparams)
        self._scope = scope
        add_arg_scope(tf.layers.conv2d_transpose)

    @property
    def scope(self):
        return self._scope

    def __call__(self, inputs):
        # Hyperparameters
        num_filters = self.hparams.num_filters

        with tf.variable_scope(self.scope, values=[inputs], reuse=tf.AUTO_REUSE):
            with arg_scope([tf.layers.conv2d_transpose],
                           kernel_initializer=tf.initializers.random_normal(stddev=0.02),
                           bias_initializer=tf.initializers.constant()):
                inputs = linear(inputs, 7 * 7 * num_filters * 2, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.ReLU)
                inputs = tf.reshape(inputs, (-1, 7, 7, num_filters * 2))

                inputs = deconv2d(inputs, num_filters, 3, strides=2, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.ReLU,
                                  method=DeconvMethod.ConvTranspose)
                inputs = deconv2d(inputs, 1, 3, strides=2, norm_fn=Norm.NONE, non_linear_fn=Nonlinear.Sigmoid,
                                  method=DeconvMethod.ConvTranspose)

                return inputs
