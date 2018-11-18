import abc
import tensorflow as tf
from tfutils.monte_carlo_integral import (
    MonteCarloIntegral, monte_carlo_integrate)
from tfutils.graph import get_dependent_variables


class GANLoss(object):

  def __init__(self, discriminator, generator):
    self.discriminator = discriminator
    self.generator = generator


class BaseVanillaGAN(abc.ABC):
  r"""
  ```math
  Let $\mathcal{X}$ the space of data (ambient-space), and $\mathcal{Z}$
  the latent, equipped with measurement $P_Z$ (prior). Discriminator

  $$D: \mathcal{X} \mapsto \mathbb{R}.$$

  And generator

  $$G: \mathcal{Z} \mapsto \mathcal{X}.$$

  The loss is defined as

  \begin{equatin}

    L[D, G] :=
      + \mathbb{E}_{x \sim P_{\textrm{data}}} \left[ \ln \sigma(D(x)) \right]
      + \mathbb{E}_{z \sim P_{Z}} \left[
            \ln \left( 1 - \sigma(D(G(z))) \right)
        \right].

  \end{equation}

  Maximizing $L$ when training $D$ and minimizing when training $G$.

  However, for numerical efficiency and stability, the loss when training $G$
  is re-defined as

  \begin{equatin}

    L[G \mid D] :=
      - \mathbb{E}_{z \sim P_{Z}} \left[
            \ln \sigma(D(G(z))) -
            \ln \left( 1 - \sigma(D(G(z))) \right)
        \right],

  \end{equation}

  instead, for any $D$ given. This re-definition keeps the global miminum
  invariant.
  ```

  Notations:
    A: Event-shape of the ambient.
    L: Event-shape of the latent.

  Args:
    n_samples: Positive integer. Shall be greater than 30 as a thumb-rule
      for central limit theorem, employed by the Monte-Carlo integral
      (i.e. the E_{z~Pz}[...]).
  """

  def __init__(self, n_samples=32):
    self.n_samples = n_samples

    self._generator_vars = None
    self._discriminator_vars = None

  def loss(self,
           data,
           name='GAN_loss',
           reuse=tf.AUTO_REUSE):
    r"""Follows the "Alternative M-step #2" in reference [1].

    ```math
    Denote sigmoid function by $\sigma$ and softplus by $s$. Useful relations:

    \begin{align}

      \ln \sigma(x) = -s(-x);

      \ln \left[ 1 - \sigma(x) \right] = -softplus(x).

    \end{align}
    ```

    References:
      1. https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/  # noqa: E501

    Args:
      data:
      name: String.
      reuse: Boolean.
    """
    with tf.name_scope(name):
      discr_real = self.discriminator(data, reuse)
      fake_data = self._generator(self.n_samples, reuse)
      discr_fake = self.discriminator(fake_data, reuse)

      self._discriminator_vars = get_dependent_variables(discr_real)
      self._generator_vars = get_dependent_variables(fake_data)

      with tf.name_scope('discriminator_loss'):
        discr_loss = MonteCarloIntegral(
            value=tf.constant(0, dtype=data.dtype),
            error=tf.constant(0, dtype=data.dtype))
        for integrands in (tf.nn.softplus(-discr_real),
                           tf.nn.softplus(discr_fake)):
          mc_int = monte_carlo_integrate(integrands, axes=[0])
          discr_loss.value += mc_int.value
          discr_loss.error += mc_int.error

      with tf.name_scope('generator_loss'):
        # gen_loss_integrand = (tf.nn.softplus(-discr_fake) -
        #                       tf.nn.softplus(discr_fake))
        gen_loss_integrand = tf.nn.softplus(-discr_fake)  # test!
        gen_loss = monte_carlo_integrate(gen_loss_integrand)

      return GANLoss(discriminator=discr_loss,
                     generator=gen_loss)

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
      # [n_samples] + A
      ambient_samples = self.generator(latent_samples, reuse)
      return ambient_samples

  @property
  def discriminator_vars(self):
    return self._discriminator_vars

  @property
  def generator_vars(self):
    return self._generator_vars
