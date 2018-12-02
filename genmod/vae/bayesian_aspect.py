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
    def experiment(n_data):
      bijector_layers = [10 for _ in range(10)]
      vae = VAE(ambient_dim=(28 * 28),
                latent_dim=64,
                bijector_layers=bijector_layers)
      bayesian_aspect = BayesianAspect(batch_size=8,
                                      vae=vae,
                                      n_data=n_data,
                                      label=3)
      bayesian_aspect.train(n_iters=20000,
                            ckpt_dir=None)
      bayesian_aspect.evaluate()
    ```

### `n_data = 8`

    ```
    ================== EVAL ENCODER ==================
    Entropy of each datum in batch:
        87.32 (0.90)
        87.15 (1.08)
        87.24 (1.03)
        86.77 (1.21)
        87.10 (0.88)
        87.12 (0.97)
        86.43 (1.03)
        87.62 (1.10)

    ================== EVAL DECODER ==================
    Entropy of each datum in batch:
        58.00 (1.47)
    ```

###  `n_data = 32`

    ```
    ================== EVAL ENCODER ==================
    Entropy of each datum in batch:
        83.96 (0.90)
        85.61 (1.08)
        85.05 (1.03)
        84.80 (1.21)
        84.42 (0.88)
        87.14 (0.97)
        84.78 (1.03)
        85.90 (1.10)

    ================== EVAL DECODER ==================
    Entropy of each datum in batch:
        60.42 (0.98)
    ```

###  `n_data = 128`

    ```
    ================== EVAL ENCODER ==================
    Entropy of each datum in batch:
        79.09 (0.90)
        75.14 (1.08)
        78.83 (1.03)
        81.75 (1.21)
        79.35 (0.88)
        80.24 (0.97)
        80.99 (1.03)
        77.72 (1.10)

    ================== EVAL DECODER ==================
    Entropy of each datum in batch:
        63.95 (1.28)
  ```

###  `n_data = 512`

    ```
    ================== EVAL ENCODER ==================
    Entropy of each datum in batch:
        79.80 (0.90)
        81.16 (1.08)
        78.60 (1.03)
        81.29 (1.21)
        79.74 (0.88)
        82.96 (0.97)
        81.29 (1.03)
        79.21 (1.10)

    ================== EVAL DECODER ==================
    Entropy of each datum in batch:
        58.11 (1.47)
    ```

###  `n_data = 2018`

    ```
    ================== EVAL ENCODER ==================
    Entropy of each datum in batch:
        76.18 (0.90)
        79.82 (1.08)
        81.03 (1.03)
        74.96 (1.21)
        76.98 (0.88)
        79.24 (0.97)
        76.16 (1.03)
        77.15 (1.10)

    ================== EVAL DECODER ==================
    Entropy of each datum in batch:
        71.39 (1.55)
    ```

Conclusion
----------
TODO
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

  def evaluate_decoder(self):
    latent_template = self.vae.prior.sample(1)
    latent = 0.1 * tf.ones_like(latent_template)
    decoder_dist = self.vae.decoder(latent, reuse=True)
    # shape: [self.batch_size]
    entropy = get_entropy(decoder_dist)

    ent_vals, ent_errs = self.sess.run([entropy.value, entropy.error])
    print('\n================== EVAL DECODER ==================\n')
    print('Entropy of each datum in batch:')
    for ent_val, ent_err in zip(ent_vals, ent_errs):
      print('    {0:.2f} ({1:.2f})'.format(ent_val, ent_err))


def std_normal_entropy():
  return 0.5 * np.log(2 * np.pi * np.exp(1))


if __name__ == '__main__':

  n_data = 8
  print('Number of data:', n_data)

  bijector_layers = [10 for _ in range(10)]
  vae = VAE(ambient_dim=(28 * 28),
            latent_dim=64,
            bijector_layers=bijector_layers)

  bayesian_aspect = BayesianAspect(batch_size=8,
                                   vae=vae,
                                   n_data=n_data,
                                   label=3)

  bayesian_aspect.train(n_iters=20000,
                        ckpt_dir=None)
  bayesian_aspect.evaluate()
