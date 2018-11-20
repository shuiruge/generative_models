import os
from tqdm import trange
from collections import namedtuple
import numpy as np
import tensorflow as tf
try:
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors
except ModuleNotFoundError:
  tfd = tf.contrib.distributions
  tfb = tfd.bijectors
from tfutils.train import (save_variables, restore_variables, ALL_VARS,
                           create_frugal_session)
from genmod.vanilla_gan_2.base import BaseVanillaGAN
from genmod.utils.mnist.data import get_dataset
from genmod.utils.mnist.plot import get_image


# Turn off TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# For reproducibility
SEED = 123
np.random.seed(SEED)
tf.set_random_seed(SEED)

# For data
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_PATH, '../../dat/')
CKPT_DIR = os.path.join(DATA_DIR, 'checkpoints/vanilla_gen')
MNIST = get_dataset(os.path.join(DATA_DIR, 'MNIST'))


class VanillaGan(BaseVanillaGAN):
  """Implements the BaseVanillaGAN by MLP.

  Comparing with: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/gan.py  # noqa: E501
  """
  def __init__(self,
               ambient_dim,
               latent_dim,
               discr_layers=(256,),
               gen_layers=(256,),
               *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.ambient_dim = ambient_dim
    self.latent_dim = latent_dim
    self.discr_layers = discr_layers
    self.gen_layers = gen_layers

  @property
  def prior(self):
    loc = tf.zeros([self.latent_dim])
    scale_diag = tf.ones([self.latent_dim])
    return tfd.MultivariateNormalDiag(loc, scale_diag)

  def generator(self, latent, reuse):
    with tf.variable_scope('generator', reuse=reuse):
      hidden = latent
      for i, layer in enumerate(self.gen_layers):
        hidden = tf.layers.dense(hidden, layer, activation=tf.nn.relu,
                                 name='hidden_layer_{}'.format(i))
        # hidden = tf.layers.dropout(hidden, rate=0.5,
        #                            name='dropout_{}'.format(i))
      logits = tf.layers.dense(hidden, self.ambient_dim, activation=None,
                               name='logits')
      return tf.sigmoid(logits)

  def discriminator(self, ambient, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
      hidden = ambient
      for i, layer in enumerate(self.discr_layers):
        hidden = tf.layers.dense(hidden, layer, activation=tf.nn.relu,
                                 name='hidden_layer_{}'.format(i))
        # hidden = tf.layers.dropout(hidden, rate=0.5,
        #                            name='dropout_{}'.format(i))
      output = tf.layers.dense(hidden, 1, activation=None,
                               name='output_layer')
      output = tf.squeeze(output, axis=-1)
      return output


Ops = namedtuple('Ops', 'data, gan_loss, discr_train_op, gen_train_op')


def get_gan_and_ops(batch_size, latent_dim):
  ambient_dim = 28 * 28  # for MNIST dataset.
  data = tf.placeholder(shape=[batch_size, ambient_dim],
                        dtype='float32', name='data')

  gan = VanillaGan(ambient_dim, latent_dim)
  gan_loss = gan.loss(data)

  print('\nGenerator variables:', gan.generator_vars, '\n')
  print('\nDiscriminator variables:', gan.discriminator_vars, '\n')

  discr_train_op = (tf.train.AdamOptimizer(learning_rate=2e-4)
                    .minimize(-gan_loss.value,
                              var_list=gan.discriminator_vars))
  gen_train_op = (tf.train.AdamOptimizer(learning_rate=2e-4)
                  .minimize(gan_loss.value,
                            var_list=gan.generator_vars))
  return gan, Ops(data, gan_loss, discr_train_op, gen_train_op)


def get_feed_dict(ops, batch_size):
  while True:
    X_batch, _ = MNIST.train.next_batch(batch_size)
    # X_batch = np.where(X_batch > 0.5,
    #                    np.ones_like(X_batch),
    #                    np.zeros_like(X_batch))
    yield {ops.data: X_batch}


def train(sess, gan, ops, feed_dict_gen, n_iters):
  sess.run(tf.global_variables_initializer())

  try:
    restore_variables(sess, ALL_VARS, CKPT_DIR)

  except Exception as e:
    print(e)

    pbar = trange(n_iters)
    for step in pbar:
      ops_to_run = [ops.discr_train_op,
                    ops.gen_train_op,
                    ops.gan_loss.value,
                    ops.gan_loss.error,
                    gan.f_divergence._peep['discriminate_part'].value,
                    gan.f_divergence._peep['discriminate_part'].error,
                    gan.f_divergence._peep['generate_part'].value,
                    gan.f_divergence._peep['generate_part'].error]
      run_result = sess.run(ops_to_run, feed_dict=next(feed_dict_gen))
      _, _, loss_val, loss_err, dp_val, dp_err, gp_val, gp_err = run_result
      pbar.set_description(
          'Loss {0:.2f} ({1:.2f}) - DP {2:.2f} ({3:.2f}) - '
          'GP {4:.2f} ({5:.2f})'.format(
              loss_val, loss_err, dp_val, dp_err, gp_val, gp_err))
      if np.isnan(loss_val):
        break

    # save_variables(sess, ALL_VARS, CKPT_DIR)


def evaluate(sess, ops, gan, n_samples):
  bernoulli_prob = gan._generator(n_samples=n_samples,
                                  reuse=tf.AUTO_REUSE)
  bernoulli_dist = tfd.Bernoulli(probs=bernoulli_prob)
  sample = bernoulli_dist.sample()
  sample_vals = sess.run(sample)
  for sample_val in sample_vals:
    image = get_image(sample_val)
    image.show()


def main(batch_size,
         latent_dim,
         n_iters,
         n_samples_to_show):
  gan, ops = get_gan_and_ops(batch_size, latent_dim)
  sess = create_frugal_session()

  feed_dict_gen = get_feed_dict(ops, batch_size)
  train(sess, gan, ops, feed_dict_gen, n_iters)

  evaluate(sess, ops, gan, n_samples_to_show)


if __name__ == '__main__':

  main(batch_size=128,
       latent_dim=100,
       n_iters=100000,
       n_samples_to_show=10)
