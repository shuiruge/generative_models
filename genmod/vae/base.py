"""
Description
-----------
Implements the abstract base class of variational auto-encoder.

Also, the lower-bound of the loss is computed, in `LossLowerBound`. However,
because of the large variance in the Monte-Carlo integral in the computation,
this lower-bound is far from usable for evaluating the performance of the
fitting.

Instead, we find that the variance of the Monte-Carlo integral of the loss is
itself an evaluation to the performance of the fitting.
"""


import abc
import tensorflow as tf
from tfutils.monte_carlo_integral import monte_carlo_integrate
try:
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  tfb = tfp.bijectors
except ImportError:
  tfd = tf.contrib.distributions
  tfb = tfd.bijectors


class BaseVAE(abc.ABC):
  """The abstract base class of "variational auto-encoder"[1].

  Notations:
    B: Batch-shape of any.
    A: Event-shape of the ambient.
    L: Event-shape of the latent.

  References:
    1. [Kingma, et al, 2014](https://arxiv.org/abs/1312.6114).

  Args:
    n_samples: Positive integer tensor-like object. Based on the thumb-rule
      of central limit theorem, it shall be no less than 30.
    name: String.
  """
  def __init__(self,
               n_samples=32,
               name='VAE'):
    self.n_samples = n_samples
    self.base_name = name

  @abc.abstractmethod
  def encoder(self, ambient, reuse):
    """Return the distribution of inference, Q(Z|x) by giving the ambient x.

    Args:
      ambient: Tensor with shape `B + A`.
      reuse: Boolean.

    Returns:
      An instance of `tfd.Distribution` with shape `B + L`.
    """
    pass

  @abc.abstractmethod
  def decoder(self, latent, reuse):
    """Returns the distribution of likelihood, P(X|z) by giving the latent z.

    Args:
      latent: Tensor of the shape `B + L`.
      reuse: Boolean.

    Returns:
      An instance of `tfd.Distribution`, with shape `B + A`.
    """
    pass

  @property
  @abc.abstractmethod
  def prior(self):
    """Returns the distribution of prior of latent variable, P(Z).

    Returns:
      An instance of `tfd.Distribution`, with shape `B + L`.
    """
    pass

  def loss(self, ambient, name='loss'):
    r"""Returns the tensors for L(X) and its error.

    Definition:
      ```math
      Denoting $X$ the ambient and $z$ the latent,

      \begin{equation}
          L(x) := E_{z \sim Q(Z \mid x)} \left[
                      \ln q(z \mid x) - \ln p(z) - \ln p(x \mid z)
                  \right].
      \end{equation}
      ```

    Evaluation:
      The performance of this fitting by minimizing the loss L(x) over
      a dataset of x can be evaluated by the variance of the Monte-Carlo
      integral in L(x). The integrand

      ```python
      p_z = ...  # the unknown likelihood distribution of latent Z.
      q_z = ...  # the inference distribution.

      z_samples = q_z.samples(N)

      for i, z in enumerate(z_samples):
          integrand[i] = q_z.log_prob(z) - (p_z.log_prob(z) + constant)
      ```

      So, the variance of the `integrand` measures the difference between the
      `q_z.log_prob` and `p_z.log_prob` in the region that the `q_z` can prob
      by sampling, regardless the `constant`. Indeed, if they are perfectly
      fitted in the probed region, then the variance vanishes, no matter what
      the value the `constant` is.

    Args:
      ambient: Tensor of the shape `batch_shape + [ambient_dim]`.
      name: String.

    Returns:
      An instance of `MonteCarloIntegral` with shape `batch_shape`.
    """
    with tf.name_scope(self.base_name):
      with tf.name_scope(name):
        # Get the distribution Q(Z|x) in definition
        encoder = self.encoder(ambient, reuse=tf.AUTO_REUSE)

        # Get the distribution P(X|z) in definition
        # [n_samples] + B + L
        latent_samples = encoder.sample(self.n_samples)
        decoder = self.decoder(latent_samples, reuse=tf.AUTO_REUSE)

        # Get the log_q(z|x) - log_p(z) - log_p(x|z) in definition
        # [n_samples] + B
        integrands = (encoder.log_prob(latent_samples) -
                      self.prior.log_prob(latent_samples) -
                      decoder.log_prob(ambient))
        return monte_carlo_integrate(integrands, axes=[0],
                                     n_samples=self.n_samples)
