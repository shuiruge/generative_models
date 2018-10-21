import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
try:
  import tensorflow_probability as tfp  # pylint: disable=E0401
  tfd = tfp.distributions
except ModuleNotFoundError:
  tfd = tf.contrib.distributions
tfb = tfd.bijectors
from tfutils.train import (save_variables, restore_variables,
                           create_frugal_session)
from tfutils.graph import get_dependent_variables
from scipy.misc import logsumexp
from vae import get_loss_X, get_log_p_X


# For reproducibility
SEED = 123
np.random.seed(SEED)
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


def logmeanexp(array, axis=None):
  if axis is None:
    n = np.sum(array.shape())
  else:
    n = np.sum(array.shape, axis=axis, keepdims=True)
  return logsumexp(array - np.log(n), axis=axis)


def main(n_iters, batch_size=128, z_dim=64, use_bijectors=True):

  # --- Build the graph ---

  X_dim = 28 * 28  # for MNIST dataset.
  X = tf.placeholder(shape=[batch_size, X_dim], dtype='float32', name='X')
  p_z = tfd.MultivariateNormalDiag(
      loc=tf.zeros([batch_size, z_dim]),
      scale_diag=tf.sqrt(tf.ones([batch_size, z_dim]) / z_dim),
      name='p_z')

  def _get_q_z_X(X, reuse):
    if use_bijectors:
      bijectors = get_bijectors(reuse=reuse)
    else:
      bijectors = []
    return get_q_z_X(X, z_dim, bijectors=bijectors, reuse=reuse)

  def _get_p_X_z(z, reuse):
    return get_p_X_z(z, X_dim=X_dim, reuse=reuse)

  n_samples = tf.placeholder(shape=[], dtype='int32', name='n_samples')
  loss_X = get_loss_X(_get_q_z_X, _get_p_X_z, p_z, n_samples=n_samples)
  loss_X_mcint = loss_X(X)
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
    for step in range(n_iters):
        n_sample_val = 1 if step < 1000 else 128
        X_batch, _ = mnist.train.next_batch(batch_size)
        feed_dict = {X: X_batch, n_samples: n_sample_val}
        sess.run([train_op, loss_X_scalar], feed_dict)
    save_variables(sess, None, os.path.join(DATA_DIR, 'checkpoints/vae'))
  
  # --- Evaluation ---

  log_p_X = get_log_p_X(_get_p_X_z, p_z,
                        n_samples=n_samples,
                        reuse=tf.AUTO_REUSE)
  log_p_X_mcint = log_p_X(X)

  X_batch, _ = mnist.test.next_batch(batch_size)
  feed_dict = {X: X_batch, n_samples: 128}

  loss_vals = sess.run(loss_X_mcint.value, feed_dict)
  loss_error_vals = sess.run(loss_X_mcint.error, feed_dict)
  log_p_X_vals = sess.run(log_p_X_mcint.value, feed_dict)
  log_p_X_error_vals = sess.run(log_p_X_mcint.error, feed_dict)

  for i, loss_val in enumerate(loss_vals):
    loss_error_val = loss_error_vals[i]
    log_p_X_val = log_p_X_vals[i]
    log_p_X_error_val = log_p_X_error_vals[i]
    print('loss V.S. -ln p(X)  ---  {0:.2f} ({1:.2f})  ---  {2:.2f} ({3:.2f})'
          .format(loss_val, loss_error_val, -log_p_X_val, log_p_X_error_val))


if __name__ == '__main__':

    main(n_iters=5000)
