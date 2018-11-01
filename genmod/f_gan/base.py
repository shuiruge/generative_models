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
    f_div: An instance of f-divergene that inherits the `BaseFDivergence`.
  """

  def __init__(self, f_div, name='f_GAN'):
    self.f_div = f_div
    self.name = name

  @property
  @abc.abstractmethod
  def generative_dist(self):
    """
    Returns:
      An instance of `Distribution` with the same batch-shape and event-shape
        as the data to be generated.
    """
    pass

  @abc.abstractmethod
  def discriminator(self, ambient, reuse):
    """
    Args:
      ambient: Tensor with the same shape as the data to be generated.
      reuse: Boolean.

    Returns:
      An 1-dimensional tensor with the same batch-shape as the data to be
      generated.
    """
    pass

  def loss(self, data):
    """
    Args:
      data: Tensor.

    Returns:
      An instance of scalar `MonteCarloIntegral`.
    """
    with tf.name_scope(self.name):
      return self.f_div(data, self.generative_dist, self.discriminator)
