"""
Description
-----------
Implements the abstract base class of f-GAN.
"""

import abc
import tensorflow as tf


class BaseFGAN(abc.ABC):
  """The abstract base class of f-GAN.

  Args:
    name: String.
  """

  def __init__(self, name='f_GAN'):
    self.name = name

  @property
  @abc.abstractmethod
  def f_divergance(self):
    """Returns an instance of f-divergene that inherits the abstract base class
    `BaseFDivergence`."""
    pass

  @property
  @abc.abstractmethod
  def prior(self):
    """The prior of the latent.

    Returns:
      Distribution.
    """
    pass

  @abc.abstractmethod
  def generator(self, latent, reuse):
    """Returns the distribution P(X|Z). where X the ambient and Z the latent.

    Args:
      latent: Tensor with a 1D batch-shape.
      reuse: Boolean.

    Returns:
      Distribution with the same batch-shape as the `latent`, and the same
      event-shape as the data to be generated.
    """
    pass

  @abc.abstractmethod
  def discriminator(self, ambient, reuse):
    """
    Args:
      ambient: Tensor with shape `[B1] + E`, for arbitrary positive
        integer `B1`, and `E` the event-shape of the data to be generated,
        and with dtype the same as the data to be generated.
      reuse: Boolean.

    Returns:
      Tensor with shape `[B1]`.
    """
    pass

  def _generator(self, n_samples, reuse):
    """
    Args:
      n_samples: Positive integer.
      reuse: Boolean.

    Returns:
      Tensor with shape `[n_samples] + E`, where `E` the event-shape of the
      data to be generated.
    """
    with tf.name_scope('generator'):
      latent_samples = self.prior.sample(n_samples)  # [n_samples] + L
      generator_dist = self.generator(latent_samples, reuse=reuse)
      ambient_samples = generator_dist.sample()  # [n_samples] + E
      return ambient_samples

  def loss(self, data):
    """
    Args:
      data: Tensor.

    Returns:
      An instance of scalar `MonteCarloIntegral`.
    """
    with tf.name_scope(self.name):
      return self.f_divergance(data, self._generator, self.discriminator)
