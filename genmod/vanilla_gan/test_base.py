import os
from tqdm import trange
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
from genmod.vanilla_gan.base import BaseVanillaGAN
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
MNIST = get_dataset(os.path.join(DATA_DIR, 'MNIST'))


class VanillaGan(BaseVanillaGAN):
  """Implements the BaseVanillaGAN by MLP.

  Comparing with: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/gan.py  # noqa: E501
  """
  def __init__(self,
               ambient_dim,
               latent_dim,
               discr_layers=(256,),
               gen_layers=(256, 512),
               **kwargs):
    super().__init__(**kwargs)
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


class TestVanillaGAN(object):

  def __init__(self,
               batch_size,
               vanilla_gan,
               discr_optimizer=None,
               gen_optimizer=None):
    self.batch_size = batch_size
    assert vanilla_gan.ambient_dim == (28 * 28)  # for MNIST.
    self.vanilla_gan = vanilla_gan
    if discr_optimizer is None:
      self.discr_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    else:
      self.discr_optimizer = discr_optimizer
    if gen_optimizer is None:
      self.gen_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    else:
      self.gen_optimizer = gen_optimizer

    self.build_graph()
    self.sess = create_frugal_session()

  def build_graph(self):
    self.data = tf.placeholder(
        shape=[self.batch_size, self.vanilla_gan.ambient_dim],
        dtype='float32', name='data')

    self.loss = self.vanilla_gan.loss(self.data)

    self.discr_train_op = self.discr_optimizer.minimize(
        loss=-self.loss.value,
        var_list=self.vanilla_gan.discriminator_vars)
    self.gen_train_op = self.gen_optimizer.minimize(
        loss=self.loss.value,
        var_list=self.vanilla_gan.generator_vars)

  def train(self, n_iters, ckpt_dir=None):
    self.sess.run(tf.global_variables_initializer())

    if ckpt_dir is not None:
      try:
        restore_variables(self.sess, ALL_VARS, ckpt_dir)
      except Exception as e:
        print(e)

    self.train_body(n_iters)

    if ckpt_dir is not None:
      save_variables(self.sess, ALL_VARS, ckpt_dir)

  def train_body(self, n_iters):
    pbar = trange(n_iters)
    for step in pbar:
      X_batch, _ = MNIST.train.next_batch(self.batch_size)
      feed_dict = {self.data: X_batch}
      self.sess.run(self.discr_train_op, feed_dict)
      self.sess.run(self.gen_train_op, feed_dict)
      loss_val, loss_err = self.sess.run(
          [self.loss.value, self.loss.error],
          feed_dict)
      pbar.set_description('Loss {0:.2f} ({1:.2f})'
                           .format(loss_val, loss_err))
      if np.isnan(loss_val):
        break

  def evaluate(self, n_samples_to_show):
    bernoulli_prob = self.vanilla_gan._generator(
        n_samples=n_samples_to_show,
        reuse=tf.AUTO_REUSE)
    bernoulli_dist = tfd.Bernoulli(probs=bernoulli_prob)
    sample = bernoulli_dist.sample()
    sample_vals = self.sess.run(sample)
    for sample_val in sample_vals:
      image = get_image(sample_val)
      image.show()

  def main(self, n_iters, ckpt_dir=None):
    self.train(n_iters, ckpt_dir)
    self.evaluate()


if __name__ == '__main__':

  vanilla_gan = VanillaGan(ambient_dim=(28 * 28),
                           latent_dim=64)

  test_case = TestVanillaGAN(batch_size=128,
                             vanilla_gan=vanilla_gan)

  test_case.main(n_iters=20000)
