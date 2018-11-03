import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
try:
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors
except ModuleNotFoundError:
  tfd = tf.contrib.distributions
  tfb = tfd.bijectors
from tfutils.train import (save_variables, restore_variables, ALL_VARS,
                           create_frugal_session)
from genmod.vanilla_gan import BaseVanillaGAN


# For reproducibility
SEED = 123
np.random.seed(SEED)
tf.set_random_seed(SEED)

# For data
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_PATH, '../../dat/')
CKPT_DIR = os.path.join(DATA_DIR, 'checkpoints/vanilla_gen')


class VanillaGan(BaseVanillaGAN):

  def __init__(self,
               ambient_dim,
               latent_dim,
               discr_layers=(128, 128),
               gen_layers=(128, 128),
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

  def generator(self, latent, reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
      hidden = latent
      for i, layer in enumerate(self.gen_layers):
        hidden = tf.layers.dense(hidden, layer, activation=tf.nn.relu,
                                 name='hidden_layer_{}'.format(i))
      logits = tf.layers.dense(hidden, self.ambient_dim, activation=None,
                               name='logits')
      return tfd.Bernoulli(logits=logits)

  def discriminator(self, ambient, reuse=None):
    with tf.variable_scope('discriminator', reuse=reuse):
      hidden = ambient
      for i, layer in enumerate(self.discr_layers):
        hidden = tf.layers.dense(hidden, layer, activation=tf.nn.relu,
                                 name='hidden_layer_{}'.format(i))
      output = tf.layers.dense(hidden, 1, activation=None,
                               name='output_layer')
      output = tf.squeeze(output, axis=-1)
      return output


def main(batch_size=128, n_iters=int(1e+5)):

  # --- Build the graph ---

  ambient_dim = 28 * 28  # for MNIST dataset.
  data = tf.placeholder(shape=[batch_size, ambient_dim],
                        dtype='float32', name='data')

  vanilla_gan = VanillaGan(ambient_dim, latent_dim=8)
  loss = vanilla_gan.loss(data)

  # TODO

  # --- Training ---

  sess = create_frugal_session()
  sess.run(tf.global_variables_initializer())

  mnist = input_data.read_data_sets(
      os.path.join(DATA_DIR, 'MNIST'),
      one_hot=True,
      source_url='http://yann.lecun.com/exdb/mnist/')

  try:
    restore_variables(sess, ALL_VARS, CKPT_DIR)

  except Exception as e:
    print(e)

    pbar = trange(n_iters)
    # TODO
    save_variables(sess, ALL_VARS, CKPT_DIR)


if __name__ == '__main__':

  main()
