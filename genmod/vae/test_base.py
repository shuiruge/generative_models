r"""
Description
-----------
Test the `BaseVAE` on MNIST dataset. The encoder and decoder are both
multilayer-perceptrons. Bijectors in the inference are employed.

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
try:
  import tensorflow_probability as tfp  # pylint: disable=E0401
  tfd = tfp.distributions
  tfb = tfp.bijectors
except ModuleNotFoundError:
  tfd = tf.contrib.distributions
  tfb = tfd.bijectors
from tfutils.pyutils import inheritdocstring
from tfutils.train import save_variables, restore_variables, ALL_VARS
from tfutils.tensorboard import variable_summaries
from genmod.vae.base import BaseVAE


# For reproducibility
SEED = 123
np.random.seed(SEED)  # pylint: disable=E1101
tf.set_random_seed(SEED)

# For data
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_PATH, '../../dat/')


def get_iafs(layers, name='bjiectors', reuse=None):
  """Complexify the inference distribution by extra-bijectors like
  normalizing flows (IAFs herein).

  Args:
    layers: List of positive integers.
    name: String.
    reuse: Boolean.

  Returns:
    List of `Bijector`s.
  """
  with tf.variable_scope(name, reuse=reuse):
    bijectors = []
    # return bijectors  # test!
    for layer in layers:
      # Get one bijector
      shift_and_log_scale_fn = \
          tfb.masked_autoregressive_default_template([layer])
      # MAP is extremely slow in training. Use IAF instead.
      bijector = tfb.Invert(
          tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn))
      bijectors.append(bijector)
    return bijectors


@inheritdocstring
class VAE(BaseVAE):

  def __init__(self,
               ambient_dim,
               latent_dim,
               decoder_layers=(128, 256, 512),
               encoder_layers=(512, 256, 128),
               bijectors=(),
               **kwargs):
    super().__init__(**kwargs)

    self.ambient_dim = ambient_dim
    self.latent_dim = latent_dim
    self.decoder_layers = decoder_layers
    self.encoder_layers = encoder_layers
    self.bijectors = bijectors

  def encoder(self, ambient, reuse):
    with tf.variable_scope('encoder', reuse=reuse):
      hidden = ambient
      for i, hidden_layer in enumerate(self.encoder_layers):
        with tf.name_scope('hidden_layer_{}'.format(i)):
          hidden = tf.layers.dense(hidden, hidden_layer)
          hidden = tf.layers.batch_normalization(hidden)
          hidden = tf.nn.relu(hidden)
          variable_summaries(hidden)
      with tf.name_scope('output_layer'):
        # Outputs in the fiber-bundle space
        output = tf.layers.dense(hidden, self.latent_dim * 2,
                                 activation=None)
      # shape: [batch_size, z_dim]
      mu, log_var = tf.split(
          output, [self.latent_dim, self.latent_dim],
          axis=1)
      variable_summaries(mu, name='summaries_mu')
      variable_summaries(log_var, name='summaries_log_var')

      base_dist = tfd.MultivariateNormalDiag(mu, tf.exp(log_var))
      chain = tfb.Chain(self.bijectors)
      encoder_dist = tfd.TransformedDistribution(base_dist, chain)
      return encoder_dist

  def decoder(self, latent, reuse):
    with tf.variable_scope('decoder', reuse=reuse):
      hidden = latent
      for i, hidden_layer in enumerate(self.decoder_layers):
        with tf.name_scope('hidden_layer_{}'.format(i)):
          hidden = tf.layers.dense(hidden, hidden_layer)
          hidden = tf.layers.batch_normalization(hidden)
          hidden = tf.nn.relu(hidden)
          variable_summaries(hidden)
      with tf.name_scope('logits'):
        logits = tf.layers.dense(hidden, self.ambient_dim, activation=None)

      decoder_dist = tfd.Bernoulli(logits=logits)
      # Make the event-shape `[X_dim]`,
      # and leaving all left axes as batch-shape
      decoder_dist = tfd.Independent(decoder_dist,
                                     reinterpreted_batch_ndims=1)
      return decoder_dist

  @property
  def prior(self):
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([self.latent_dim]),
        scale_diag=tf.ones([self.latent_dim]),
        name='prior')


def process_X(X_batch):
  """Makes the pixal of the image either `0` or `1`."""
  return np.where(X_batch > 0.5,
                  np.ones_like(X_batch),
                  np.zeros_like(X_batch))


def generate_random_X(batch_size):
  return np.random.uniform(size=[batch_size, 28 * 28])


def add_noise(X_batch, noise_ratio):
  # Add noise
  noise = np.random.uniform(low=-noise_ratio, high=noise_ratio,
                            size=X_batch.shape)
  X_batch = X_batch + noise
  # Bound it
  X_batch = np.where(X_batch > 1.0, np.ones_like(X_batch), X_batch)
  X_batch = np.where(X_batch < 0.0, np.zeros_like(X_batch), X_batch)
  return X_batch


class TestVAE(object):
  """
  Args:
    batch_size: Positive integer.
    vae: A `VAE` instance.
    batch_generator: Generator that generates data-batch.
    optimizer: A `tf.train.Optimizer` instance.
    logdir: String.
    sess_config: A `tf.ConfigProto` instance.
  """

  def __init__(self,
               batch_size,
               vae,
               batch_generator,
               optimizer=None,
               logdir=None,
               sess_config=None):
    self.batch_size = batch_size
    self.vae = vae
    self.batch_generator = batch_generator

    if optimizer is None:
      self.optimizer = tf.train.AdamOptimizer(epsilon=1e-3)
    else:
      self.optimizer = optimizer

    self.build_graph()
    self.sess = tf.Session(config=sess_config)

    self.logdir = logdir
    if self.logdir is not None:
      self.summary_writer = tf.summary.FileWriter(
          self.logdir, self.sess.graph)

  def build_graph(self):
    self.data = tf.placeholder(
        shape=[self.batch_size, self.vae.ambient_dim],
        dtype='float32', name='data')
    self.n_samples = tf.placeholder(shape=[], dtype='int32', name='n_samples')

    self.loss = self.vae.loss(self.data)
    self.loss_scalar = tf.reduce_mean(self.loss.value)
    variable_summaries(self.loss_scalar, name='summaries_loss')

    optimizer = tf.train.AdamOptimizer(epsilon=1e-3)
    self.train_op = optimizer.minimize(self.loss_scalar)
    self.summary_op = tf.summary.merge_all()

    with tf.name_scope('global_step'):
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
      self.global_step_inscr = self.global_step.assign_add(1)

  def train(self, n_iters, skip_step, ckpt_dir):
    """
    Args:
      n_iters: Non-negative integer.
      skip_step: Positive integer.
      ckpt_dir: String or `None`.

    Returns:
      List of floats.
    """
    self.sess.run(tf.global_variables_initializer())

    if ckpt_dir is not None:
      try:
        restore_variables(self.sess, ALL_VARS, ckpt_dir)
      except Exception as e:
        print(e)

    loss_vals = self.train_body(n_iters, skip_step)

    if ckpt_dir is not None:
      save_variables(self.sess, ALL_VARS, ckpt_dir)

    return loss_vals

  def train_body(self, n_iters, skip_step):
    """
    Args:
      n_iters: Non-negative integer.
      skip_step: Positive integer.

    Returns:
      List of floats.
    """
    loss_vals = []
    pbar = trange(n_iters)
    for step in pbar:
      # n_sample_val = 1 if step < (n_iters // 2) else 128
      n_sample_val = 1  # faster for doing experiments.
      X_batch = next(self.batch_generator)
      feed_dict = {self.data: X_batch,
                   self.n_samples: n_sample_val}
      _, loss_val = self.sess.run([self.train_op, self.loss_scalar],
                                  feed_dict)
      loss_vals.append(loss_val)
      smeared_loss_val = np.mean(loss_vals[-300:])
      pbar.set_description('loss (smeared): {0:.2f}'
                           .format(smeared_loss_val))

      # Tensorboard summary
      if (self.logdir is not None) and (step % skip_step == 0):
        summary_vals = self.sess.run(self.summary_op, feed_dict)
        global_step_val = self.sess.run(self.global_step)
        self.summary_writer.add_summary(
            summary_vals, global_step=global_step_val)

      self.sess.run(self.global_step_inscr)
    return loss_vals

  def evaluate(self):
    X_batch = next(self.batch_generator)
    self._evaluate(X_batch)

    # Noised data
    noised_X_batch = add_noise(X_batch, noise_ratio=0.8)
    self._evaluate(noised_X_batch)

    # Fake data
    fake_X_batch = generate_random_X(batch_size=128)
    self._evaluate(fake_X_batch)

  def _evaluate(self, data_batch):
    """
    Args:
      data_batch: Numpy array.
    """
    feed_dict = {self.data: data_batch, self.n_samples: 128}
    loss_vals = self.sess.run(self.loss.value, feed_dict)
    loss_error_vals = self.sess.run(self.loss.error, feed_dict)

    for i, loss_val in enumerate(loss_vals):
      loss_error_val = loss_error_vals[i]
      print('loss: {0:.2f} ({1:.2f})'.format(loss_val, loss_error_val))

  def main(self, n_iters, skip_step, ckpt_dir=None):
    """
    Args:
      n_iters: Non-negative integer.
      skip_step: Positive integer.
      ckpt_dir: String or `None`.
    """
    self.train(n_iters, skip_step, ckpt_dir)
    self.evaluate()


if __name__ == '__main__':

  from genmod.utils.mnist.data import get_dataset

  mnist = get_dataset(os.path.join(DATA_DIR, 'MNIST'))

  def get_batch_generator(batch_size):
    while True:
      X_batch, _ = mnist.train.next_batch(batch_size)
      yield process_X(X_batch)

  batch_size = 128
  batch_generator = get_batch_generator(batch_size)

  bijectors = []
  # bijectors = get_iafs([10 for _ in range(10)])

  vae = VAE(ambient_dim=(28 * 28),
            latent_dim=64,
            bijectors=bijectors)

  test_case = TestVAE(batch_size, vae, batch_generator)

  test_case.main(n_iters=20000, skip_step=50)
