import tensorflow as tf
from tfutils.monte_carlo_integral import monte_carlo_integrate
from genmod.f_gan.base import BaseFGAN
from genmod.f_gan.f_divergence import BaseFDivergence


def log_sigmoid(x):
  r"""Returns `ln[ sigmoid(x) ]`.

  ```math
  Denote sigmoid-function by $\sigma$ and softplus-function by $s$, then

  \begin{equation}
    \ln \sigma(x) = -s(-x).
  \end{equation}
  ```
  """
  return -tf.nn.softplus(-x)


def log1m_sigmoid(x):
  r"""Returns `ln[ 1 - sigmoid(x) ]`.

  ```math
  Denote sigmoid-function by $\sigma$ and softplus-function by $s$, then

  \begin{equation}
    \ln \left( 1 - \sigma(x) \right) = -s(x).
  \end{equation}
  ```
  """
  return -tf.nn.softplus(x)


class GANDivergence(BaseFDivergence):
  r"""Follows the "Alternative M-step #2" in reference [1].

  ```math
  Denote sigmoid function by $\sigma$ and softplus by $s$. GAN-divergence
  (inheriting f-divergence) is defined by setting

  \begin{align}
    g_f(x) := \ln \sigma(x);
    f^{\*}(x) := x - \ln \left( 1 - e^x \right).
  \end{align}

  Denote the discriminator as $D: \mathcal{X} \mapsto \mathbb{R}$, and
  $V := \sigma \circ D, \mathcal{X} \mapsto (0, 1)$. Then, by the relations

  \begin{align}
    \ln \sigma(x) = -s(-x);
    \ln \left[ 1 - \sigma(x) \right] = -s(x),
  \end{align}

  the GAN-divergence can be reduced to

  \begin{equation}
    D_{\textrm{GAN}} \left[ P \| Q \right]:=
      \mathbb{E}_{x \sim P} \left[ \ln V(x) \right] -
      \mathbb{E}_{x \sim Q} \left[ \ln \frac{ V(x) }{ 1 - V(x) } \right],
  \end{equation}

  wherein the first term is the "discriminate part" and the second the
  "generate part".
  ```

  References:
    1. https://www.inference.vc/an-alternative-update-rule-for-generative-adversarial-networks/  # noqa: E501

  Args:
    n_samples: Positive integer.
    name: String.
  """

  def __init__(self, n_samples, name='GAN_divergence'):
    super().__init__(n_samples=n_samples, name=name)

  @staticmethod
  def output_activation(x):
    return log_sigmoid(x)

  @staticmethod
  def f_star(x):
    return x - tf.log1p(-tf.exp(x))

  def generate_part(self, fake_data, discrimator, reuse):
    """Re-implementing for avoiding the numerical instability caused by
    the `tf.exp` in `self.f_star`."""
    discr_fake = discrimator(fake_data, reuse)
    integrands = -log_sigmoid(discr_fake) + log1m_sigmoid(discr_fake)
    return monte_carlo_integrate(integrands, axes=[0])


class BaseVanillaGAN(BaseFGAN):

  def __init__(self, n_samples=32, **kwargs):
    super().__init__(**kwargs)
    self.n_samples = n_samples
    self.gan_divergence = GANDivergence(self.n_samples)

  @property
  def f_divergence(self):
    return self.gan_divergence
