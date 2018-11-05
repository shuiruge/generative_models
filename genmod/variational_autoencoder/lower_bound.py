"""
Description
-----------
Implements the `LossLowerBound` of variational auto-encoder. However, because
of its intrinsic large variance, this implementation is not to be employed,
but staying here for keeping remind when the Monte-Carlo integral fails.
"""


import tensorflow as tf
from tfutils.monte_carlo_integral import MonteCarloIntegral


class LossLowerBound:
  r"""The function ln p(x) by Monte-Carlo integral, which is the lower bound
  of the loss `LossX`.

  ```math
  p(x) = E_{z \sim P(Z)} \left[ p(X \mid z) \right].
  ```

  The error of the Monte-Carlo integral is computed as follow.

  ```math
  \begin{equation}
    \delta \ln p(x) = \frac{ \delta p(x) }{ p(x) }.
  \end{equation}
  ```

  wherein

  ```math
  \begin{align}
    & \left( \frac{ \delta p(x) }{ p(x) } \right)^2 \\
    = & \frac{
          \text{Var}_{z \sim P(Z)} \left[ p(x \mid z) \right]
        }{
          \text{E}_{z \sim P(Z)}^2 \left[ p(x \mid z) \right]
        } \\
    = & \frac{
          \text{E}_{z \sim P(Z)}^2 \left[ p^2(x \mid z) \right]
        }{
          \text{E}_{z \sim P(Z)}^2 \left[ p(x \mid z) \right]
        } - 1.
  \end{align}
  ```

  WARNING:
    This estimation of lower bound of the fitting by the KL-divergence is
    NOT proper, because of its large variance.

    Indeed, as the number of samples in the Monte-Carlo integral increases,
    the variance increases, rather than decreasing as what should be expected.
    This is caused by the large variance of P(X|Z), which is a multiplication
    of P(X_i|Z)s where each X_i is for one pixal of the `28 * 28`-pixal
    picture of the MNIST dataset. (Say, the multiplication of `28 * 28`
    independent probabilities all with the value `0.9`, `0.9 ** (28 * 28)`,
    is extremely tiny.)

  Args:
    get_p_X_z: Callable with the signature:
      Args:
        z: Tensor of the shape `batch_shape + [z_dim]`.
        reuse: Boolean.
      Returns:
        An instance of `tfd.Distribution`, with batch-shape `batch_shape`
        and event-shape `[ambient_dim]`.
    p_z: An instance of `tfd.Distribution`, with batch-shape `batch_size`
      and event-shape `z_dim`.
    n_samples: Positive integer.
    reuse: If reuse the variables in `get_p_X_z`.
  """

  def __init__(self,
               variational_autoencoder,
               epsilon=1e-8,
               name='LossLowerBound'):
    self.vae = variational_autoencoder
    self.epsilon = epsilon
    self.base_name = name

    self.log_n_samples = tf.log(
        tf.cast(self.vae.n_samples, self.vae.prior.dtype),
        name='log_n_samples')

  def log_expectation(self, log_samples, name='log_expectation'):
    """ln E[ exp(log_samples) ]

    Args:
      log_samples: Tensor of the shape `[n_samples]` + batch-shape +
        event-shape.

    Returns:
      Tensor of the shape batch_shape + event-shape.
    """
    with tf.name_scope(name):
      return tf.reduce_logsumexp(log_samples - self.log_n_samples, axis=0)

  def __call__(self, ambient):
    """Returns the tensor of ln p(X). This serves as the lower bound of
    the loss by KL-divergence, evaluating the fitting.

    Args:
      ambient: Tensor of the shape `batch_shape + [ambient_dim]`.
      name: String.

    Returns:
      An instance of `MonteCarloIntegral` with shape `batch_shape`.
    """
    with tf.name_scope(self.base_name):
      # [n_samples] + batch_shape + [latent_dim]
      latent_samples = self.vae.prior.sample(self.vae.n_samples)
      decoder = self.vae.decoder(latent_samples, reuse=tf.AUTO_REUSE)
      # [n_samples] + batch_shape
      decoder_log_probs = decoder.log_prob(ambient)

      # E_{z~p(z)} [ p(X|z) ]
      # batch_shape
      lower_bound_tensor = self.log_expectation(decoder_log_probs,
                                                name='lower_bound')

      # Error of Monte-Carlo integral
      square_delta_lower_bound = -1.0 + tf.exp(self.log_expectation(
          2.0 * (decoder_log_probs - lower_bound_tensor)))
      delta_lower_bound = tf.sqrt(square_delta_lower_bound,
                                  name='delta_lower_bound')

      return MonteCarloIntegral(lower_bound_tensor, delta_lower_bound)
