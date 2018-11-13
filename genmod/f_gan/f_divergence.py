"""
Description
-----------
Implements the abstract base class of f-divergence.
"""


import abc
import tensorflow as tf
from tfutils.monte_carlo_integral import MonteCarloIntegral


class BaseFDivergence(abc.ABC):
  r"""Abstract base class for f-divergence.

  Definition:
    ```math
    Let $f$ a convex function on $\mathbb{R}$, with domain $\textrm{dom}_f$
    in \mathbb{R}$. Let $g$ a function $\mathbb{R} \mapsto \textrm{dom}_f$,
    called {\it output activation function}. Let $P$ the empirical distribution
    of the data and $Q$ the distribution fitting to $P$, by minimizing the
    {\it f-divergence} defined as

    \begin{equation}
        D_f := \mathbb{E}_{x \sim P} \left[ g(D(x)) \right] +
               \mathbb{E}_{x \sim Q} \left[ f^{\*}( g(D(x)) ) \right].
    \end{equation}
    ```

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

  @abc.abstractmethod
  def output_activation(self, x):
    r"""The output activation function

    ```math
    g : \mathbb{R} \mapsto \textrm{dom}_{f^{\*}}
    ```

    Args:
      x: Tensor with shape `[?]`.

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
      x: Tensor with shape `[?]`.

    Returns:
      Tensor with the same shape as `x`.
    """
    pass

  def __call__(self,
               data,
               generator,
               discriminator,
               reuse=tf.AUTO_REUSE):
    """
    Args:
      data: Tensor with shape `[B] + E`, for arbitrary positive integer `B` and
        tuple of positive integers `E`. The data shall be unbiased and the `B`
        shall be greater than 30, as a thumb-rule for central limit theorem,
        employed by the Monte-Carlo integral (i.e. the E_{x~P}[...]).
      generator: Callable with signature:
        Args:
          n_samples: Positive integer.
          reuse: Boolean.
        Returns:
          Tensor with shape `[n_samples] + E`.
      discriminator: Callable with the signature:
        Args:
          ambient: Tensor with shape `[B1] + E`, for arbitrary positive
            integer `B1`, and with dtype the same as `data`.
          reuse: Boolean.
        Returns:
          Tensor with shape `[B1]`.
      reuse: Boolean.

    Returns:
      An instance of scalar `MonteCarloIntegral`.

    Raises:
      EventShapeError.
    """
    with tf.name_scope(self.name):

      def _discriminator(ambient):
        return discriminator(ambient, reuse=reuse)

      def _generator(n_samples):
        return generator(n_samples, reuse=reuse)

      # E_{x~P} [ g(D(x)) ]
      mc_integrand = self.output_activation(_discriminator(data))  # [B]
      mean, var = tf.nn.moments(mc_integrand, axes=[0])  # []
      mc_integral_val = mean
      batch_size, *rests = data.get_shape().as_list()
      mc_integral_err = tf.sqrt(var / tf.cast(batch_size, dtype=var.dtype))

      # x ~ Q
      ambient_samples = _generator(self.n_samples)  # [self.n_samples] + E
      self.check_same_event_shape(data, ambient_samples)

      # E_{x~Q} [ -f*( g(D(x)) ) ]
      mc_integrand = - self.f_star(
          self.output_activation(
              _discriminator(ambient_samples)))  # [self.n_samples]
      mean, var = tf.nn.moments(mc_integrand, axes=[0])  # []
      mc_integral_val += mean
      n_samples = tf.cast(self.n_samples, dtype=var.dtype)
      mc_integral_err += tf.sqrt(var / n_samples)

      return MonteCarloIntegral(value=mc_integral_val,
                                error=mc_integral_err)

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
