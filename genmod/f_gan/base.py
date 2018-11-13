"""
Description
-----------
Implements the abstract base class of f-GAN.
"""

import abc
import tensorflow as tf
from tfutils.graph import get_dependent_variables


class BaseFGAN(abc.ABC):
  """The abstract base class of f-GAN.

  ```math:
  1. $D$ the dataset;
  2. $Z \in \mathcal{Z}$ the latent random variable, obeys prior $P(Z)$;
  3. $g: \mathcal{Z} \mapsto \mathcal{X}$ the generator-function;
  4. $X \in \mathcal{X}$ the ambient random variable, by $X = g(Z)$;
  5. $d: \mathcal{X} \mapsto \mathbb{R}$ the discriminator-function.
  ```

  Notations:
    A: Event-shape of the ambient.
    L: Event-shape of the latent.

  Args:
    name: String.
  """

  def __init__(self, name='f_GAN'):
    self.name = name

    # Initialize as `None`s
    self._generator_vars = None
    self._discriminator_vars = None

  @property
  @abc.abstractmethod
  def f_divergance(self):
    """Returns an instance of f-divergene that inherits the abstract base class
    `BaseFDivergence`."""
    pass

  @property
  @abc.abstractmethod
  def prior(self):
    """The prior distribution of the latent.

    Returns:
      Distribution with event-shape `L`.
    """
    pass

  @abc.abstractmethod
  def generator(self, latent, reuse):
    """The function `g` that gives X = g(Z), where X the ambient and Z
    the latent.

    Args:
      latent: Tensor with shape `[B] + L`, for arbitrary positive integer `B`.
      reuse: Boolean.

    Returns:
      Tensor with shape `[B] + A`.
    """
    pass

  @abc.abstractmethod
  def discriminator(self, ambient, reuse):
    """
    Args:
      ambient: Tensor with shape `[B] + A`, for arbitrary positive integer
        `B`.
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
      Tensor with shape `[n_samples] + A`.
    """
    with tf.name_scope('generator'):
      # [n_samples] + L
      latent_samples = self.prior.sample(n_samples)
      # [n_samples] + E
      ambient_samples = self.generator(latent_samples, reuse=reuse)

      # The `tf.Variable`s that the generator depends on,
      # for the property `self.generator_vars`
      self._generator_vars = get_dependent_variables(ambient_samples)

      return ambient_samples

  def loss(self, data, reuse=tf.AUTO_REUSE):
    """
    Args:
      data: Tensor.

    Returns:
      An instance of scalar `MonteCarloIntegral`.
    """
    with tf.name_scope(self.name):
      loss_mc_int = self.f_divergance(
          data, self._generator, self.discriminator, reuse)

      # The `tf.Variable`s that the generator depends on,
      # for the property `self.generator_vars`
      all_vars = get_dependent_variables(loss_mc_int.value)
      self._discriminator_vars = [
          _ for _ in all_vars if _ not in self.generator_vars]

      return loss_mc_int

  @property
  def generator_vars(self):
    """Returns a list of `tf.Variable`s that the generator depends on."""
    return self._generator_vars

  @property
  def discriminator_vars(self):
    """Returns a list of `tf.Variable`s that the discriminator depends on."""
    return self._discriminator_vars
