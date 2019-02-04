from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tflibs.model import Model
from tflibs.training import Optimizer, Dispatcher, Optimizers

from networks.vae import Encoder, Decoder
from tflibs.ops import concat_images, normalize


class VAE(Model):
    def __init__(self, features, model_idx=0, device=None, **hparams):
        Model.__init__(self, features, model_idx=model_idx, device=device, **hparams)

        self.networks.update(encoder=Encoder(**hparams),
                             decoder=Decoder(**hparams))

    @Model.image(summary='Example_Images')
    def example_images(self):
        return self.features['image']

    @Model.tensor
    def latent_dist(self):
        return self.networks.encoder(self.example_images)

    @Model.tensor(summary='Latent_Mean')
    def latent_mean(self):
        return self.latent_dist.mean()

    @Model.tensor(summary='Latent_Std')
    def latent_std(self):
        return self.latent_dist.stddev()

    @Model.tensor(summary='Latent_Vars')
    def latent_vars(self):
        return self.latent_dist.sample()

    @Model.tensor
    def marginal(self):
        return self.networks.decoder(self.latent_vars)

    @Model.image(summary='Reconstructed_Images')
    def reconstructed_images(self):
        return tf.reshape(self.marginal, tf.shape(self.example_images))

    @Model.image
    def sampled_images(self):
        return tf.reshape(self.networks.decoder(self.features['z']), (28, 28, 1))

    @Model.image(summary='Grouped_Image')
    def images_group(self):
        return concat_images(self.example_images, self.reconstructed_images)

    @Model.loss
    def reconstruction_loss(self):
        flatten = tf.layers.flatten(self.example_images)

        return tf.losses.mean_squared_error(flatten, self.marginal, scope='Reconstruction_Loss',
                                            weights=self.hparams.data_size)

    @Model.loss(summary='KL_Divergence')
    def kld(self):
        return tf.reduce_mean(tf.reduce_sum(self.latent_dist.kl_divergence(tf.distributions.Normal(0., 1.)), axis=-1))

    @Model.loss
    def loss(self):
        return self.kld + self.reconstruction_loss

    @classmethod
    def train(cls, features: dict, learning_rate, **hparams):
        # TODO: Optimizer
        optimizer = Optimizer(learning_rate, '', optimizer=Optimizers.AdaGrad)

        dispatcher = Dispatcher(cls, hparams, features, num_towers=1, model_parallelism=False)

        train_op = dispatcher.minimize(optimizer, lambda m: m.loss, global_step=tf.train.get_or_create_global_step())

        chief = dispatcher.chief  # typd: VAE
        tf.logging.info('Explicitly declared summaries')
        for t in [chief.images_group, chief.latent_mean, chief.latent_std]:
            tf.logging.info(t)

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
            # 'reconstruction': tf.image.convert_image_dtype(chief.reconstructed_images, tf.uint8),
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

        if mode == tf.estimator.ModeKeys.TRAIN:
            model_args.update(params['train_args'])
            return VAE.train(features, **model_args)
        elif mode == tf.estimator.ModeKeys.EVAL:
            model_args.update(params['eval_args'])
            return VAE.evaluate(features, **model_args)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return VAE.predict(features, **model_args)
        else:
            raise ValueError

    @staticmethod
    def map_fn(image, label, _id):
        return {
                   'image': normalize(image),
                   '_id': _id,
               }, tf.to_float(label)

    @staticmethod
    def eval_map_fn(*args, **kwargs):
        return VAE.map_fn(*args, **kwargs)

    @classmethod
    def add_model_args(cls, argparser, parse_args):
        argparser.add_argument('--dim-z',
                               type=int,
                               default=3,
                               help='The dimension of latent `z`')
        argparser.add_argument('--num-hidden',
                               type=int,
                               default=500,
                               help='The number of hidden units')
        argparser.add_argument('--data-size',
                               type=int,
                               default=28 * 28 * 1,
                               help='The size of data')

    @classmethod
    def add_train_args(cls, argparser, parse_args):
        ############
        # Training #
        ############
        argparser.add_argument('--learning-rate',
                               type=float,
                               default=0.001,
                               help='Learning rate.')

    @classmethod
    def add_eval_args(cls, argparser, parse_args):
        argparser.add_argument('--gpu-eval',
                               type=str,
                               help='GPU ids for evaluation.',
                               default=None)


export = VAE
