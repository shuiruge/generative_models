r"""
Abstract
--------
Trying to figure out the Bayesian aspect of generative model (variational
auto-encoder herein). Explicitly, how the number of data effects the model's
confidence, and how to describe the confidence itself.

Experiements
------------
Parameters are set as follow:

    ```python
    latent_
    n_data = ...
    batch_size = ...
    label = ...
    n_iters = ...
    bijector_layers = [10 for _ in range(10)]
    vae = VAE(ambient_dim=(28 * 28),
              latent_dim=64,
              bijector_layers=bijector_layers)
    bayesian_aspect = BayesianAspect(batch_size=batch_size,
                                     vae=vae,
                                     n_data=n_data,
                                     label=label)
    bayesian_aspect.train(n_iters=n_iters,
                          ckpt_dir=None)
      bayesian_aspect.evaluate()
    ```

### `n_data = 8`

    ```
    ================== EVAL ENCODER ==================

    Entropy of each datum in batch:
        86.13 (1.07)
        85.08 (0.93)
        ...
    Mean: 86.95

    ================== EVAL DECODER ==================

    Entropy of each datum in batch:
        80.88 (1.15)
        70.04 (1.47)
        ...
    Mean: 69.85
    ```

###  `n_data = 32`

    ```
    ================== EVAL ENCODER ==================

    Entropy of each datum in batch:
        83.89 (1.07)
        84.67 (0.93)
        ...
    Mean: 84.39

    ================== EVAL DECODER ==================

    Entropy of each datum in batch:
        88.74 (1.54)
        78.72 (1.43)
        ...
    Mean: 67.35
    ```

###  `n_data = 128`

    ```
    ================== EVAL ENCODER ==================

    Entropy of each datum in batch:
        80.82 (1.07)
        80.14 (0.93)
        ...
    Mean: 80.76

    ================== EVAL DECODER ==================

    Entropy of each datum in batch:
        140.86 (1.41)
        126.01 (1.36)
        ...
    Mean: 115.74
    ```

###  `n_data = 512`

    ```
    ================== EVAL ENCODER ==================

    Entropy of each datum in batch:
        74.89 (1.07)
        76.00 (0.93)
        ...
    Mean: 76.39

    ================== EVAL DECODER ==================

    Entropy of each datum in batch:
        59.16 (1.35)
        56.49 (1.11)
        ...
    Mean: 60.44
    ```

###  `n_data = 2018`

    ```
    ================== EVAL ENCODER ==================

    Entropy of each datum in batch:
        71.69 (1.07)
        71.90 (0.93)
        ...
    Mean: 72.38

    ================== EVAL DECODER ==================

    Entropy of each datum in batch:
        72.61 (1.65)
        72.23 (1.05)
        ...
    Mean: 69.28
    ```

Conclusion
----------
* Increasing the number of data regularly varies the entropy of the
  distribution of encoder (from 87.0 to 72.4 for latent dimension 64).

* This entropy is almost linear to the `log2` of the number of data, with
  slop about `-2.0`.

* Not regularly for the entropy of the distribution of decoder.

* The final loss varies a bit around.

* Varing the latent dimension (e.g. from 64 to 128) enlarges the entropy
  of the distribution of encoder in magnitude (e.g. from 80 to 160). But
  the linear relation between this entropy and the `log2` of the number of
  data is not changed.

* Varing the latent dimension (e.g. from 64 to 128) does not manifestly
  change the entropy of the distribution of decoder in magnitude.
"""

import os
import numpy as np
import tensorflow as tf
from tfutils.monte_carlo_integral import monte_carlo_integrate
from genmod.utils.mnist.data import get_dataset
from genmod.vae.test_base import VAE, TestVAE


# For reproducibility
SEED = 123
np.random.seed(SEED)  # pylint: disable=E1101
tf.set_random_seed(SEED)

# For data
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_PATH, '../../dat/')
MNIST = get_dataset(os.path.join(DATA_DIR, 'MNIST'))


def get_entropy(distribution, n_samples=32, name='entropy'):
  """Returns the entropy of the distribution `distribution` as a Monte-Carlo
  integral.

  Args:
    distribution: A `Distribution` instance.
    n_samples: Positive integer.

  Returns:
    A `MonteCarloIntegral` instance.
  """
  with tf.name_scope(name):
    samples = distribution.sample(n_samples)
    # shape: # [n_samples] + batch-shape
    integrands = - distribution.log_prob(samples)
    # shape: batch-shape
    return monte_carlo_integrate(integrands, axes=[0])


class BayesianAspect(object):

  def __init__(self, batch_size, vae, n_data, label):
    assert batch_size <= n_data
    self.batch_size = batch_size
    self.vae = vae
    self.n_data = n_data
    self.label = label

    self.training_data = self.get_training_data(n_data, label)

    self.test_vae = TestVAE(
        batch_size=batch_size,
        vae=vae,
        batch_generator=self.get_batch_generator())
    self.sess = self.test_vae.sess
    self.data = self.test_vae.data
    self.batch_generator = self.test_vae.batch_generator
    self.train = self.test_vae.train

  def get_training_data(self, n_data, label):
    training_data = []
    for i, (X, y) in enumerate(zip(MNIST.train.images,
                                   MNIST.train.labels)):
      if np.argmax(y) == label:
        training_data.append(X)
      if len(training_data) == n_data:
        break
    training_data = np.array(training_data)
    return training_data

  def get_batch_generator(self):
    while True:
      # For shuffling without mutating `self.training_data`
      training_data = self.training_data.copy()
      np.random.shuffle(training_data)
      yield training_data[:self.batch_size]

  def evaluate(self):
    self.evaluate_encoder()
    self.evaluate_decoder()

  def evaluate_encoder(self):
    encoder_dist = self.vae.encoder(self.data, reuse=True)
    # shape: [self.batch_size]
    entropy = get_entropy(encoder_dist)

    feed_dict = {self.data: self.training_data[:self.batch_size]}
    ent_vals, ent_errs = self.sess.run([entropy.value, entropy.error],
                                       feed_dict)
    print('\n================== EVAL ENCODER ==================\n')
    print('Entropy of each datum in batch:')
    for ent_val, ent_err in zip(ent_vals, ent_errs):
      print('    {0:.2f} ({1:.2f})'.format(ent_val, ent_err))
    print('Mean: {0:.2f}'.format(np.mean(ent_vals)))

  def evaluate_decoder(self):
    latent_template = tf.ones_like(self.vae.prior.sample())
    rs = [0.1 * i for i in range(self.batch_size)]
    latent = tf.stack([r * latent_template for r in rs])
    decoder_dist = self.vae.decoder(latent, reuse=True)
    # shape: [self.batch_size]
    entropy = get_entropy(decoder_dist)

    ent_vals, ent_errs = self.sess.run([entropy.value, entropy.error])
    print('\n================== EVAL DECODER ==================\n')
    print('Entropy of each datum in batch:')
    for ent_val, ent_err in zip(ent_vals, ent_errs):
      print('    {0:.2f} ({1:.2f})'.format(ent_val, ent_err))
    print('Mean: {0:.2f}'.format(np.mean(ent_vals)))


def normal_dist_entropy(scale):
  return 0.5 * np.log(2 * np.pi * np.exp(1) * scale ** 2)


if __name__ == '__main__':

  from argparse import ArgumentParser

  parser = ArgumentParser()
  parser.add_argument('--n_data', type=int, help='Greater than batch_size.')
  parser.add_argument('--latent_dim', type=int)
  parser.add_argument('--batch_size', type=int, default=8)
  parser.add_argument('--label', type=int, default=3)
  parser.add_argument('--n_iters', type=int, default=30000)
  args = parser.parse_args()

  def do_experiment(args, bijector_layers=([10] * 10)):
    vae = VAE(
        ambient_dim=(28 * 28),
        latent_dim=args.latent_dim,
        bijector_layers=bijector_layers)
    bayesian_aspect = BayesianAspect(
        batch_size=args.batch_size,
        vae=vae,
        n_data=args.n_data,
        label=args.label)
    bayesian_aspect.train(
        n_iters=args.n_iters,
        ckpt_dir=None)
    bayesian_aspect.evaluate()

  print(args)
  do_experiment(args)
