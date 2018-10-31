"""
Description
-----------
Test the `VariationalAutoencoder` on MNIST dataset. The encoder and decoder
are both multilayer-perceptrons. Bijectors in the inference are employed.

Conclusion
----------
1. The loss approaches about 70 after 20000 iterations, with latent-dimension
   16, without further fine-tuning.
1. Also test the loss-per-datum on the test dataset of MNIST. The losses keep
   the same order as in the training stage.
1. The loss-per-datum on the noised test dataset keeps the same order as in
   the training when the noise-ratio is 50%. However, it becomes 10X greater
   when the noise-ratio is 80%.
1. The loss-per-datum on the randomly generated fake dataset is 20X greater
   than that in the training.
1. Variational auto-encoder can, thus, also be an anomaly detector, detecting
   the exotic data mixing into a given dataset.
"""


import os
from tqdm import trange
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
try:
  import tensorflow_probability as tfp  # pylint: disable=E0401
  tfd = tfp.distributions
  tfb = tfp.bijectors
except ModuleNotFoundError:
  tfd = tf.contrib.distributions
  tfb = tfd.bijectors
from tfutils.train import (save_variables, restore_variables,
                           create_frugal_session)
from tfutils.graph import get_dependent_variables
from scipy.misc import logsumexp
from genmod.variational_autoencoder.base import BaseVariationalAutoencoder


# For reproducibility
SEED = 123
np.random.seed(SEED)  # pylint: disable=E1101
tf.set_random_seed(SEED)

# For data
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_PATH, '../../dat/')


def get_p_X_z(z, X_dim, hidden_layers=None,
              name='p_X_z', reuse=None):
  """Returns the distribution of P(X|Z).
  
  X | Z ~ Bernoulli( p(Z) ).
  
  Args:
    z: Tensor of the shape `batch_shape + [z_dim]`.
    X_dim: Positive integer.
    hidden_layers: List of positive integers. Defaults to
      `[128, 256, 512]`.
    
  Returns:
    An instance of `tfd.Distribution`, with batch-shape `batch_shape`
    and event-shape `[X_dim]`.
  """
  if hidden_layers is None:
    hidden_layers = [128, 256, 512]
    
  with tf.variable_scope(name, reuse=reuse):
    hidden = z
    for hidden_layer in hidden_layers:
      hidden = tf.layers.dense(hidden, hidden_layer,
                               activation=tf.nn.relu)
    logits = tf.layers.dense(hidden, X_dim, activation=None)

    p_X_z = tfd.Bernoulli(logits=logits)
    # Make the event-shape `[X_dim]`,
    # and leaving all left axes as batch-shape
    p_X_z = tfd.Independent(p_X_z, reinterpreted_batch_ndims=1)
    return p_X_z


def get_q_z_X(X, z_dim, hidden_layers=None, bijectors=None,
              name='q_z_X', reuse=None):
  """Returns the distribution of Z | X.
  
  Z = bijector(Z_0), and
  Z_0 | X ~ Normal(mu(X;phi), sigma(X;phi)).
  
  Args:
    X: Tensor with shape `batch_shape + [X_dim]`.
    z_dim: Positive integer.
    hidden_layers: List of positive integers. Defaults to
      `[512, 256, 128]`.
    bijectors: List of `tfb.Bijector`s. Defaults to an empty
      list.
    
  Returns:
    An instance of `tfd.Distribution`, with batch-shape `batch_shape`
    and event-shape `[z_dim]`.
  """
  if bijectors is None:
    bijectors = []
  if hidden_layers is None:
    hidden_layers = [512, 256, 128]
    
  with tf.variable_scope(name, reuse=reuse):
    hidden = X
    for hidden_layer in hidden_layers:
      hidden = tf.layers.dense(hidden, hidden_layer,
                               activation=tf.nn.relu)
    # Outputs in the fiber-bundle space
    output = tf.layers.dense(hidden, z_dim * 2, activation=None)
    # shape: [batch_size, z_dim]
    mu, log_var = tf.split(output, [z_dim, z_dim], axis=1)
    
    q_z0_X = tfd.MultivariateNormalDiag(mu, tf.exp(log_var))
    chain = tfb.Chain(bijectors)
    q_z_X = tfd.TransformedDistribution(q_z0_X, chain)
    return q_z_X


def get_bijectors(name='bjiectors', reuse=None):
  """Complexify the inference distribution by extra-bijectors like
  normalizing flows.
  
  Returns:
    List of `Bijector`s.
  """
  with tf.variable_scope(name, reuse=reuse):
    bijectors = []
    #return bijectors  # test!
    for _ in range(10):
      # Get one bijector
      shift_and_log_scale_fn = \
        tfb.masked_autoregressive_default_template([128])
      # MAP is extremely slow in training. Use IAF instead.
      bijector = tfb.Invert(
          tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn))
      bijectors.append(bijector)
    return bijectors


class VariationalAutoencoder(BaseVariationalAutoencoder):

  def __init__(self, batch_size, X_dim, z_dim, use_bijectors, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.batch_size = batch_size
    self.X_dim = X_dim
    self.z_dim = z_dim
    self.bijectors = get_bijectors() if use_bijectors else []

  def encoder_dist(self, X, reuse):
    return get_q_z_X(X, self.z_dim, bijectors=self.bijectors, reuse=reuse)

  def decoder_dist(self, z, reuse):
    return get_p_X_z(z, X_dim=self.X_dim, reuse=reuse)

  @property
  def prior(self):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([self.batch_size, self.z_dim]),
        scale_diag=tf.ones([self.batch_size, self.z_dim]),
        name='p_z')


def process_X(X_batch):
  """Makes the pixal of the image either `0` or `1`."""
  return np.where(X_batch > 0.5,
                  np.ones_like(X_batch),
                  np.zeros_like(X_batch))


def generate_random_X(batch_size):
  return np.random.uniform(size=[batch_size, 28*28])


def add_noise(X_batch, noise_ratio):
  # Add noise
  noise = np.random.uniform(low=-noise_ratio, high=noise_ratio,
                            size=X_batch.shape)
  X_batch = X_batch + noise
  # Bound it
  X_batch = np.where(X_batch > 1.0, np.ones_like(X_batch), X_batch)
  X_batch = np.where(X_batch < 0.0, np.zeros_like(X_batch), X_batch)
  return X_batch


def main(n_iters, batch_size=128, z_dim=16, use_bijectors=True):

  # --- Build the graph ---

  X_dim = 28 * 28  # for MNIST dataset.
  X = tf.placeholder(shape=[batch_size, X_dim], dtype='float32', name='X')
  n_samples = tf.placeholder(shape=[], dtype='int32', name='n_samples')
  vae = VariationalAutoencoder(batch_size, X_dim, z_dim, use_bijectors,
                               n_samples=n_samples)

  loss_X_mcint = vae.loss(X)
  loss_X_scalar = tf.reduce_mean(loss_X_mcint.value)

  optimizer = tf.train.AdamOptimizer(epsilon=1e-3)
  train_op = optimizer.minimize(loss_X_scalar)

  # --- Training ---

  sess = create_frugal_session()
  sess.run(tf.global_variables_initializer())

  mnist = input_data.read_data_sets(
      os.path.join(DATA_DIR, 'MNIST'),
      one_hot=True,
      source_url='http://yann.lecun.com/exdb/mnist/')

  try:
    restore_variables(sess, None, os.path.join(DATA_DIR, 'checkpoints/vae'))

  except Exception as e:
    print(e)
    
    pbar = trange(n_iters)
    for step in pbar:
        n_sample_val = 1 if step < (n_iters // 2) else 128
        X_batch, _ = mnist.train.next_batch(batch_size)
        feed_dict = {X: process_X(X_batch), n_samples: n_sample_val}
        _, loss_val = sess.run([train_op, loss_X_scalar], feed_dict)
        pbar.set_description('loss: {0:.2f}'.format(loss_val))

    save_variables(sess, None, os.path.join(DATA_DIR, 'checkpoints/vae'))
  
  # --- Evaluation ---

  X_batch, _ = mnist.test.next_batch(batch_size)
  feed_dict = {X: process_X(X_batch), n_samples: 128}
  loss_vals = sess.run(loss_X_mcint.value, feed_dict)
  loss_error_vals = sess.run(loss_X_mcint.error, feed_dict)

  for i, loss_val in enumerate(loss_vals):
    loss_error_val = loss_error_vals[i]
    print('loss: {0:.2f} ({1:.2f})'.format(loss_val, loss_error_val))

  # Noised data
  X_batch, _ = mnist.test.next_batch(batch_size)
  X_batch = add_noise(X_batch, noise_ratio=0.8)
  feed_dict = {X: process_X(X_batch), n_samples: 128}
  loss_vals = sess.run(loss_X_mcint.value, feed_dict)
  loss_error_vals = sess.run(loss_X_mcint.error, feed_dict)

  for i, loss_val in enumerate(loss_vals):
    loss_error_val = loss_error_vals[i]
    print('noised loss: {0:.2f} ({1:.2f})'.format(loss_val, loss_error_val))

  # Fake data
  fake_X_batch = generate_random_X(batch_size=128)
  feed_dict = {X: process_X(fake_X_batch), n_samples: 128}
  loss_vals = sess.run(loss_X_mcint.value, feed_dict)
  loss_error_vals = sess.run(loss_X_mcint.error, feed_dict)

  for i, loss_val in enumerate(loss_vals):
    loss_error_val = loss_error_vals[i]
    print('fake loss: {0:.2f} ({1:.2f})'.format(loss_val, loss_error_val))


if __name__ == '__main__':

    main(n_iters=20000)
