import tensorflow as tf
from genmod.f_gan import BaseFDivergence, BaseFGAN


class KLDivergence(BaseFDivergence):

  def output_activation(self, x):
    r"""Returns a negative tensor.

    ```math
    \begin{equation}
        g(x) := - \ln \left( 1 + \exp{-x} \right).
    \end{equation}
    ```
    """
    return -1.0 * tf.nn.softplus(-1.0 * x)

  def f_star(self, x):
    r"""Returns a positive tensor.

    ```math
    \begin{equation}
        f^{\*}(x) := - \ln \left( 1 - \exp{x} \right).
    \end{equation}
    ```
    """
    return -1.0 * tf.log1p(-1.0 * tf.exp(x))


class BaseVanillaGAN(BaseFGAN):

  def __init__(self, n_samples=32, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.n_samples = n_samples
    self.kl_divergance = KLDivergence(self.n_samples)

  @property
  def f_divergance(self):
    return self.kl_divergance
