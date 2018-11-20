"""
Description
-----------
Implements the abstract base class of f-divergence.
"""


import abc
import tensorflow as tf
from tfutils.monte_carlo_integral import monte_carlo_integrate


class BaseFDivergence(abc.ABC):
  r"""Abstract base class for f-divergence.

  Definition:
    ```math
    Denotes:
      $f^{\*}$: Convex function on $\mathbb{R}$, with domain
        $\textrm{dom}_{f^{\*}$;
      $g_f$: Function $\mathbb{R} \mapsto \textrm{dom}_{f^{\*}}$, called
        "output activation function";
      $P$: The empirical distribution of the data;
      $Q$: The distribution fitting to $P$, by minimizing the f-divergence

    F-divergence $D_f \left( P \| Q \right)$ is defined as

    \begin{equation}

      D_f := \mathbb{E}_{x \sim P} \left[ g_f(D(x)) \right] +
             \mathbb{E}_{x \sim Q} \left[ f^{\*}( g_f(D(x)) ) \right],

    \end{equation}

    where the first line is called "discriminator part" and the second line
    "generator part".
    ```

  Notations:
    A: The event-shape of the ambient.

  References:
    1. [Nowozin, et al. (2016)](https://arxiv.org/abs/1606.00709).

  Args:
    n_samples: Positive integer. Shall be greater than 30 as a thumb-rule
      for central limit theorem, employed by the Monte-Carlo integral
      (i.e. the E_{x~Q}[...]).
    name: String.
  """

  def __init__(self, n_samples=32, name='f_divergence'):
    self.n_samples = n_samples
    self.name = name

    # For peering
    self._peep = {}

  @abc.abstractmethod
  def output_activation(self, x):
    r"""The output activation function

    ```math
    g_f: \mathbb{R} \mapsto \textrm{dom}_{f^{\*}}
    ```

    Args:
      x: Tensor with shape `[None]`.

    Returns:
      Tensor with the same shape as `x`.
    """
    pass

  @abc.abstractmethod
  def f_star(self, x):
    r"""The Legendre-Fenchel conjugate of some given function `f`.

    Definition:
      ```math
      For $\forall f$ given, its Legendre-Fenchel conjugate is defined as

      \begin{equation}
         f^{\*}(k) := \sup_{k} \left\{ k x - f(x) \right\}.
      \end{equation}
      ```

    Args:
      x: Tensor with shape `[None]`.

    Returns:
      Tensor with the same shape as `x`.
    """
    pass

  def discriminate_part(self, data, discriminator, reuse):
    """Returns the `E_{x~P} [ g_f(D(x)) ]`.

    Args:
      data: Tensor with shape `[None] + A`.
      discriminator: Callable with the signature:
        Args:
          ambient: Tensor with shape `[None] + A`.
          reuse: Boolean.
        Returns:
          Tensor with the same batch-shape as the `ambient`.
      reuse: Boolean.

    Returns:
      A scalar `MonteCarloIntegral` instance.
    """
    with tf.name_scope('discriminator_part'):
      # [B]
      integrands = self.output_activation(
          discriminator(data, reuse))
      return monte_carlo_integrate(integrands, axes=[0])

  def generate_part(self, fake_data, discriminator, reuse):
    """Returns the `E_{x~Q} [ -f*( g_f(D(x)) ) ]`.

    Args:
      fake_data: Tensor with shape `[None] + A`.
      discriminator: Callable with the signature:
        Args:
          ambient: Tensor with shape `[None] + A`.
          reuse: Boolean.
        Returns:
          Tensor with the same batch-shape as the `ambient`.
      reuse: Boolean.

    Returns:
      A scalar `MonteCarloIntegral` instance.
    """
    with tf.name_scope('generator_part'):
      # [self.n_samples]
      integrands = - self.f_star(
          self.output_activation(
              discriminator(fake_data, reuse)))
      return monte_carlo_integrate(integrands, axes=[0])

  def __call__(self,
               data,
               discriminator,
               generator,
               reuse=tf.AUTO_REUSE):
    """
    Args:
      data: Tensor with shape `[None] + A`.
      discriminator: Callable with the signature:
        Args:
          ambient: Tensor with shape `[None] + A`.
          reuse: Boolean.
        Returns:
          Tensor with the same batch-shape as the `ambient`.
      generator: Callable with signature:
        Args:
          n_samples: Positive integer.
          reuse: Boolean.
        Returns:
          Tensor with shape `[n_samples] + A`.
      reuse: Boolean.

    Returns:
      A scalar `MonteCarloIntegral` instance.

    Raises:
      EventShapeError.
    """
    with tf.name_scope(self.name):
      # [self.n_samples] + E
      fake_data = generator(self.n_samples, reuse)
      self.check_same_event_shape(data, fake_data)

      discr_part = self.discriminate_part(data, discriminator, reuse)
      gen_part = self.generate_part(fake_data, discriminator, reuse)

      # Store as extra information
      self._peep['discriminate_part'] = discr_part
      self._peep['generate_part'] = gen_part

      return discr_part + gen_part

  def check_same_event_shape(self, data, ambient_samples):
    """
    Args:
      data: Tensor.
      ambient_samples: Tensor.

    Raises:
      EventShapeError: If `data` and `ambient_samples` do not share the same
        event-shape.
    """
    if get_event_shape(data) != get_event_shape(ambient_samples):
      raise EventShapeError('Data "{0}" and ambient-samples "{1}" should '
                            'share the same event-shape.')


def get_event_shape(x):
  """
  Args:
    x: Tensor of the shape `[B] + E` where `B` is the batch_size, and `E` the
      event-shape.

  Returns:
    The event-shape `E`, as a list of positive integers.
  """
  batch_shape, *event_shape = shape_list(x)
  return event_shape


def shape_list(tensor):
  """Returns the shape of the tensor `tensor` as a list.

  Args:
    tensor: Tensor.

  Returns:
    List of positive integers.
  """
  return tensor.get_shape().as_list()


class EventShapeError(Exception):
  pass
