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
    n_samples: Positive integer.
    name: String.
  """

  def __init__(self, n_samples, name='f_divergence'):
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

  def __call__(self, data, generative_dist, discriminator):
    """
    Args:
      data: Tensor.
      generative_dist: Distribution with the same batch-shape and event-shape
        as the `data`.
      discriminator: Callable with the signature:
        Args:
          ambient: Tensor with the same shape as the `data`.
          reuse: Boolean.
        Returns:
          An 1-dimensional tensor with the same batch-shape as the `data`.

    Returns:
      An instance of scalar `MonteCarloIntegral`.
    """
    with tf.name_scope(self.name):

      def _discriminator(ambient):
        return discriminator(ambient, reuse=tf.AUTO_REUSE)

      # Initialize for `MonteCarloIntegral`
      mc_int = MonteCarloIntegral(value=0.0, error=0.0)

      # E_{x~P} [ g(D(x)) ]
      mc_integrand = self.output_activation(_discriminator(data))  # [B]
      mean, var = tf.nn.moments(mc_integrand, axes=[0])
      mc_int.value += mean
      batch_size, *rests = data.get_shape().aslist()
      mc_int.error += tf.sqrt(var / tf.cast(batch_size, dtype=var.dtype))

      # E_{x~Q} [ -f*( g(D(x)) ) ]
      samples = generative_dist.sample(self.n_samples)
      mc_integrand = -1.0 * self.f_star(
          self.output_activation(_discriminator(samples)))  # [B]
      mean, var = tf.nn.moments(mc_integrand, axes=[0])
      mc_int.value += mean
      mc_int.error += tf.sqrt(var / tf.cast(self.n_samples, dtype=var.dtype))

      return mc_int
