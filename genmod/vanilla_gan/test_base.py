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
from genmod.vanilla_gan import BaseVanillaGAN
from genmod.utils.mnist.data import get_dataset


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


Ops = namedtuple('Ops', 'data, loss, max_train_op, min_train_op')


def get_gan_and_ops(batch_size):
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

  return gan, Ops(data, loss, max_train_op, min_train_op)


def get_feed_dict(ops, batch_size):
  while True:
    X_batch, _ = MNIST.train.next_batch(batch_size)
    X_batch = np.where(X_batch > 0.5,
                       np.ones_like(X_batch),
                       np.zeros_like(X_batch))
    yield {ops.data: X_batch}


def get_train_op(ops, n_iters_per_epoch):
  train_op = ops.min_train_op  # initialize.
  step = 0
  while True:
    if (step + 1) % n_iters_per_epoch == 0:  # switch train-op
      if train_op == ops.max_train_op:
        train_op = ops.min_train_op
      else:
        train_op = ops.max_train_op
    yield train_op


def train(sess, ops, train_op_gen, feed_dict_gen, n_iters):
  sess.run(tf.global_variables_initializer())

  try:
    restore_variables(sess, ALL_VARS, CKPT_DIR)

  except Exception as e:
    print(e)

    pbar = trange(n_iters)
    for _ in pbar:
      _, loss_val, loss_err = sess.run(
          [next(train_op_gen), ops.loss.value, ops.loss.error],
          feed_dict=next(feed_dict_gen))
      pbar.set_description('Loss {0:.2f} ({1:.2f})'
                           .format(loss_val, loss_err))

    save_variables(sess, ALL_VARS, CKPT_DIR)


def evaluate(sess, ops, gan):
  bernoulli_prob = gan._generator(n_samples=100, reuse=tf.AUTO_REUSE)
  bernoulli_dist = tfd.Bernoulli(probs=bernoulli_prob)
  sample = bernoulli_dist.sample()
  sample_vals = sess.run(sample)
  return sample_vals


def main(batch_size=128, n_epoches=100, n_iters=int(1e+4)):
  gan, ops = get_gan_and_ops(batch_size)
  sess = create_frugal_session()

  n_iters_per_epoch = int(n_iters / n_epoches)
  train_op_gen = get_train_op(ops, n_iters_per_epoch)
  feed_dict_gen = get_feed_dict(ops, batch_size)
  train(sess, ops, train_op_gen, feed_dict_gen, n_iters)

  evaluate(sess, ops, gan)


if __name__ == '__main__':

  main()
