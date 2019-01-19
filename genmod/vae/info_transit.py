r"""
Description
-----------
Experiments for declaring the transition of information between normalizing-
flows, being in the middle of a VAE.

Conclusion
----------
* Within all trialed parameters, setting `n_bijectors = 5` and `scale = 1e-0`
  gains the best performance, with loss lower than that without normalizing-
  flow by about `5.0`.

* Decreasing `scale` does not lead to a smaller final smeared loss in the case
  where `n_bijectors = 5`.

* Enlarging `n_bijectors`, e.g. to `10` but keeping `scale = 1e-0` leads to
  numerical instability at the beginning of training. Decreasing `scale` to,
  e.g., `1e-1` solves this problem.

* The reason why decreasing `scale` does not make better performance (except
  for better numerical stability) maybe that the normalizing-flow keeps the
  information in the transiting invariant. This is a significant conclusion.
  The "information" (by Shannon) does characterize the information of the data
  (including the latent) in the flow. So, unlike the case of ResNet, decreasing
  the `scale` do not increasing the transparentibility of the flow of the
  information of data (including the latent).
"""


import os
import numpy as np
import tensorflow as tf
try:
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors
except ImportError:
  tfd = tf.contrib.distributions
  tfb = tfd.bijectors
from tfutils.initializer import GlorotInitializer
from tfutils.distribution import get_jensen_shannon
from tfutils.pyutils import inheritdocstring
from tfutils.tensorboard import variable_summaries
from genmod.vae.test_base import VAE, process_X, TestVAE
from genmod.utils.mnist.data import get_dataset


# For reproducibility
SEED = 123
np.random.seed(SEED)  # pylint: disable=E1101
tf.set_random_seed(SEED)

# For data
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_PATH, '../../dat/')


def get_bijectors(
        layers,
        kernel_initializer=None,
        name='bjiectors',
        reuse=None):
  """Complexify the inference distribution by extra-bijectors like
  normalizing flows (IAFs herein).

  Args:
    layers: List of positive integers.
    kernel_initializer: An `Initializer` instance.
    name: String.
    reuse: Boolean.

  Returns:
    List of `Bijector`s.
  """
  with tf.variable_scope(name, reuse=reuse):
    bijectors = []

    for layer in layers:
      # MAP is extremely slow in training. Use IAF instead.
      iaf = tfb.Invert(
          tfb.MaskedAutoregressiveFlow(
              tfb.masked_autoregressive_default_template(
                  hidden_layers=[layer],
                  kernel_initializer=kernel_initializer)))
      batch_norm = tfb.BatchNormalization(training=False)

      bijectors.append(iaf)
      bijectors.append(batch_norm)
    return bijectors


def get_base_dist(encoder_dist, bijectors):
  """Returns the base distribution of the encoder distribution `encoder_dist`
  as a transformed distrubtion, which is gained by the bijectors `bijectors`.

  Args:
    encoder_dist: A distribution instance.
    bijectors: List of bijectors.

  Returns:
    A distribution instance.
  """
  inv_bijectors = [tfb.Invert(_) for _ in bijectors[::-1]]
  inv_chain = tfb.Chain(bijectors=inv_bijectors)
  base_dist = tfd.TransformedDistribution(
      distribution=encoder_dist, bijector=inv_chain)
  return base_dist


@inheritdocstring
class TestInfoTransit(TestVAE):

  def build_graph(self):
    super().build_graph()

    self.encoder_dist = self.vae.encoder(self.data, reuse=tf.AUTO_REUSE)
    self.base_dist = get_base_dist(self.encoder_dist, self.vae.bijectors)

    if self.vae.bijectors:
      jensen_shannon = get_jensen_shannon(
          p=self.encoder_dist, q=self.base_dist, n_samples=32)
      variable_summaries(jensen_shannon.value,
                         name='summaries_Jensen_Shannon')
    self.summary_op = tf.summary.merge_all()

  def main(self, n_iters, skip_step, ckpt_dir=None):
    self.train(n_iters, skip_step, ckpt_dir)


def main(args):
  mnist = get_dataset(os.path.join(DATA_DIR, 'MNIST'))

  def get_batch_generator(batch_size):
    while True:
      X_batch, _ = mnist.train.next_batch(batch_size)
      yield process_X(X_batch)

  batch_generator = get_batch_generator(args.batch_size)

  bijectors = get_bijectors(
      layers=[args.n_perceptrons for _ in range(args.n_bijectors)],
      kernel_initializer=GlorotInitializer(scale=args.scale))

  vae = VAE(ambient_dim=(28 * 28),
            latent_dim=args.latent_dim,
            # Make it shallow and easy to train
            decoder_layers=[int(_) for _ in args.decoder_layers.split(',')],
            encoder_layers=[int(_) for _ in args.encoder_layers.split(',')],
            bijectors=bijectors)

  logdir = os.path.join(DATA_DIR, args.log)

  sess_config = tf.ConfigProto()
  if args.gpus:
    sess_config.gpu_options.visible_device_list = args.gpus

  test_case = TestInfoTransit(args.batch_size, vae, batch_generator,
                              logdir=logdir, sess_config=sess_config)

  test_case.main(n_iters=args.n_iters, skip_step=args.skip_step)


if __name__ == '__main__':

  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('--n_bijectors', type=int, default=5)
  parser.add_argument('--scale', type=float, default=1e-0)
  parser.add_argument('--log', type=str, default='')
  parser.add_argument('--n_iters', type=int, default=100000)
  parser.add_argument('--skip_step', type=int, default=100)
  parser.add_argument('--batch_size', type=int, default=128)
  parser.add_argument('--latent_dim', type=int, default=64)
  parser.add_argument('--n_perceptrons', type=int, default=128,
                      help='Number of perceptrons per bijector.')
  parser.add_argument('--decoder_layers', type=str, default='128,256,512')
  parser.add_argument('--encoder_layers', type=str, default='512,256,128')
  parser.add_argument('--gpus', type=str, default='')
  args = parser.parse_args()

  main(args)
