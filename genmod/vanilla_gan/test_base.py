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

  def generator(self, latent, reuse):
    with tf.variable_scope('generator', reuse=reuse):
      hidden = latent
      for i, layer in enumerate(self.gen_layers):
        hidden = tf.layers.dense(hidden, layer, activation=tf.nn.relu,
                                 name='hidden_layer_{}'.format(i))
      logits = tf.layers.dense(hidden, self.ambient_dim, activation=None,
                               name='logits')
      return tf.sigmoid(logits)

  def discriminator(self, ambient, reuse):
    with tf.variable_scope('discriminator', reuse=reuse):
      hidden = ambient
      for i, layer in enumerate(self.discr_layers):
        hidden = tf.layers.dense(hidden, layer, activation=tf.nn.relu,
                                 name='hidden_layer_{}'.format(i))
      output = tf.layers.dense(hidden, 1, activation=None,
                               name='output_layer')
      output = tf.squeeze(output, axis=-1)
      return output


def main(batch_size=128, n_epoches=100, n_iters=int(1e+4)):

  # --- Build the graph ---

  ambient_dim = 28 * 28  # for MNIST dataset.
  data = tf.placeholder(shape=[batch_size, ambient_dim],
                        dtype='float32', name='data')

  gan = VanillaGan(ambient_dim, latent_dim=8)
  loss = gan.loss(data)

  print('\nGenerator variables:', gan.generator_vars, '\n')
  print('\nDiscriminator variables:', gan.discriminator_vars, '\n')

  max_train_op = tf.train.AdamOptimizer(epsilon=1e-3).minimize(
      -loss.value, var_list=gan.discriminator_vars)
  min_train_op = tf.train.AdamOptimizer(epsilon=1e-3).minimize(
      loss.value, var_list=gan.generator_vars)

  # --- Training ---

  sess = create_frugal_session()
  sess.run(tf.global_variables_initializer())

  mnist = input_data.read_data_sets(
      os.path.join(DATA_DIR, 'MNIST'),
      one_hot=True,
      source_url='http://yann.lecun.com/exdb/mnist/')

  def get_feed_dict():
    while True:
      X_batch, _ = mnist.train.next_batch(batch_size)
      X_batch = np.where(X_batch > 0.5,
                         np.ones_like(X_batch),
                         np.zeros_like(X_batch))
      yield {data: X_batch}

  def get_train_op():
    train_op = min_train_op
    step = 0
    n_iters_per_epoch = int(n_iters / n_epoches)
    while True:
      if (step + 1) % n_iters_per_epoch == 0:  # switch train-op
        train_op = min_train_op if train_op == max_train_op else max_train_op
      yield train_op

  try:
    restore_variables(sess, ALL_VARS, CKPT_DIR)

  except Exception as e:
    print(e)

    train_op = get_train_op()
    feed_dict = get_feed_dict()
    pbar = trange(n_iters)
    for _ in pbar:
      _, loss_val, loss_err = sess.run(
          [next(train_op), loss.value, loss.error],
          feed_dict=next(feed_dict))
      pbar.set_description('Loss {0:.2f} ({1:.2f})'
                           .format(loss_val, loss_err))

    save_variables(sess, ALL_VARS, CKPT_DIR)


if __name__ == '__main__':

  main()
