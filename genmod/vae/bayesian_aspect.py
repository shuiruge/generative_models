"""
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
  bayesian_aspect.train(n_iters=10000,
                        ckpt_dir=None)
  # ... other codes
```

1. When `n_data = 8`
    - the entropy of encoder-distribution is about 85.0 (0.8);
    - the entropy of decoder-distribution is about 47.0 (1.2).

2. When `n_data = 1024`
    - the entropy of encoder-distribution is about 77.5 (0.8);
    - the entropy of decoder-distribution is about 135 (1.6).

Conclusion
----------
Increasing the number of data, without changing the number of labels, will
decrease the entropy of the encoder-distribution by 10%, increasing the
confidence of encoding. But increase the entropy of the decoder-distribution
by 200%, enlarging the varity of the generated samples.

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
    while len(training_data) < n_data:
      X_batch, y_batch = MNIST.train.next_batch(1)
      X, y = X_batch[0], y_batch[0]
      if np.argmax(y) == label:
        training_data.append(X)
    training_data = np.array(training_data)
    return training_data

  def get_batch_generator(self):
    while True:
      np.random.shuffle(self.training_data)
      yield self.training_data[:self.batch_size]

  def evaluate(self):
    self.evaluate_encoder()
    self.evaluate_decoder()

  def evaluate_encoder(self):
    encoder_dist = self.vae.encoder(self.data, reuse=True)
    # shape: [self.batch_size]
    entropy = get_entropy(encoder_dist)

    feed_dict = {self.data: next(self.test_vae.batch_generator)}
    ent_vals, ent_errs = self.sess.run([entropy.value, entropy.error],
                                       feed_dict)
    print('\n================== EVAL ENCODER ==================\n')
    print('Entropy of each datum in batch:')
    for ent_val, ent_err in zip(ent_vals, ent_errs):
      print('    {0:.2f} ({1:.2f})'.format(ent_val, ent_err))

  def evaluate_decoder(self):
    latent = self.vae.prior.sample(1)
    decoder_dist = self.vae.decoder(latent, reuse=True)
    # shape: [self.batch_size]
    entropy = get_entropy(decoder_dist)

    ent_vals, ent_errs = self.sess.run([entropy.value, entropy.error])
    print('\n================== EVAL DECODER ==================\n')
    print(decoder_dist)
    print('Entropy of each datum in batch:')
    for ent_val, ent_err in zip(ent_vals, ent_errs):
      print('    {0:.2f} ({1:.2f})'.format(ent_val, ent_err))


if __name__ == '__main__':

  bijector_layers = [10 for _ in range(10)]
  vae = VAE(ambient_dim=(28 * 28),
            latent_dim=64,
            bijector_layers=bijector_layers)

  bayesian_aspect = BayesianAspect(batch_size=8,
                                   vae=vae,
                                   n_data=8,
                                   label=3)

  bayesian_aspect.train(n_iters=10000,
                        ckpt_dir=None)
  bayesian_aspect.evaluate()
