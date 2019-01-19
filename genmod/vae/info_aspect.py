r"""
Description
-----------

Conclusion
----------
* The maximum of entropy of a (28, 28) pixal single channel picture is
  :math:`28 \times 28 \times \log_2(256) = 6272`.

* The entropy of the distribution of encoder is about 67 (+-3). So, the rate
  of compression is about 93. The latent dimension is set to 64.

* The entropy of the distribution of decoder is about 50 (+-20). Being smaller
  than that of the encoder, as expected (data inequality).

* `H(X, Z)` and `H(X)` can then be estimated, about 110 and 40 respectively.

* Enlarging the latent dimension from 64 to 128 enlarges the entropy of the
  distribution of encoder, as expected, but, surprisingly, keeps that of the
  decoder invariant, even for the random error (entropy as a Monte-Carlo
  integral has random error).
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
from tfutils.distribution import get_entropy
from tfutils.pyutils import inheritdocstring
from tfutils.tensorboard import variable_summaries
from genmod.vae.test_base import VAE, process_X, TestVAE
from genmod.utils.mnist.data import get_dataset
from genmod.vae.info_transit import get_bijectors


# For reproducibility
SEED = 123
np.random.seed(SEED)  # pylint: disable=E1101
tf.set_random_seed(SEED)

# For data
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_PATH, '../../dat/')
MNIST = get_dataset(os.path.join(DATA_DIR, 'MNIST'))


@inheritdocstring
class TestInfoTransit(TestVAE):

  def build_graph(self):
    super().build_graph()

    encoder_dist = self.vae.encoder(self.data, reuse=tf.AUTO_REUSE)

    encoder_dist_ent = get_entropy(encoder_dist)
    variable_summaries(encoder_dist_ent.value,
                       name='entropy/encoder')

    latent = self.vae.prior.sample(32)
    decoder_dist = self.vae.decoder(latent, reuse=tf.AUTO_REUSE)
    decoder_dist_ent = get_entropy(decoder_dist)
    variable_summaries(decoder_dist_ent.value,
                       name='entropy/decoder')

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

  logdir = os.path.join(DATA_DIR, args.log) if args.log else None

  sess_config = tf.ConfigProto()
  if args.gpus:
    sess_config.gpu_options.visible_device_list = args.gpus

  test_case = TestInfoTransit(args.batch_size, vae, batch_generator,
                              logdir=logdir, sess_config=sess_config)

  ckpt_dir = args.ckpt_dir if args.ckpt_dir else None
  test_case.main(n_iters=args.n_iters,
                 skip_step=args.skip_step,
                 ckpt_dir=ckpt_dir)


if __name__ == '__main__':

  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('--n_bijectors', type=int, default=5)
  parser.add_argument('--scale', type=float, default=1e-0)
  parser.add_argument('--log', type=str, default='')
  parser.add_argument('--ckpt_dir', type=str, default='')
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
