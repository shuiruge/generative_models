import abc
import numpy as np
import tensorflow as tf
from tfutils.monte_carlo_integral import MonteCarloIntegral
try:
  import tensorflow_probability as tfp  # pylint: disable=E0401
  tfd = tfp.distributions
except:
  tfd = tf.contrib.distributions
tfb = tfd.bijectors


class BaseVariationalAutoencoder(abc.ABC):
  """The interface of variational auto-encoder.

  Args:
    n_samples: Positive integer tensor-like object. Based on the golden-rule
      of statistics, it shall be no less than 30.
    name: String.
  """
  def __init__(self,
               n_samples=32,
               name='VariationalAutoencoder'):
    self.n_samples = n_samples
    self.base_name = name

  @abc.abstractmethod
  def encoder_dist(self, ambient, reuse):
    """Return the distribution of inference, q(z|X) by giving the ambient X.

    Args:
      ambient: Tensor with shape `batch_shape + [ambient_dim]`.
      reuse: Boolean.

    Returns:
      An instance of `tfd.Distribution`, with batch-shape `batch_shape`
      and event-shape `[z_dim]`.
    """
    pass

  @abc.abstractmethod
  def decoder_dist(self, latent, reuse):
    """Returns the distribution of likelihood, p(X|z) by giving the latent z.

    Args:
      latent: Tensor of the shape `batch_shape + [latent_dim]`.
      reuse: Boolean.

    Returns:
      An instance of `tfd.Distribution`, with batch-shape `batch_shape`
      and event-shape `[ambient_dim]`.
    """
    pass

  @property
  @abc.abstractmethod
  def prior(self):
    """Returns the distribution of prior of latent variable, p(z).

    Returns:
      An instance of `tfd.Distribution`, with batch-shape `batch_shape`
      and event-shape `[latent_dim]`.
    """
    pass

  def loss(self, ambient, name='loss'):
    r"""Returns the tensors for L(X) and its error.
  
    Definition:
      ```math
      Denoting $X$ the ambient and $z$ the latent,

      \begin{equation}
          L(X) := E_{z \sim q(z \mid X)} \left[
                      \ln q(z \mid X) - \ln p(z) - \ln p(X \mid z)
                  \right].
      \end{equation}
      ```

    Evaluation:
      The performance of this fitting by minimizing the loss L(X) over
      a dataset of X can be evaluated by the variance of the Monte-Carlo
      integral in L(X). The integrand
      
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
        # Get the distribution q(z|X) in definition
        encoder_dist = self.encoder_dist(ambient, reuse=tf.AUTO_REUSE)
        
        # Get the distribution p(X|z) in definition
        # [n_samples] + batch_shape + [latent_dim]
        latent_samples = encoder_dist.sample(self.n_samples)
        decoder_dist = self.decoder_dist(latent_samples, reuse=tf.AUTO_REUSE)

        # Get the log_q(z|X) - log_p(z) - log_p(X|z) in definition
        # [n_samples] + batch_shape
        mc_samples = (encoder_dist.log_prob(latent_samples) -
                      self.prior.log_prob(latent_samples) -
                      decoder_dist.log_prob(ambient))

        # Get the Monte-Carlo integral in definition
        mean, variance = tf.nn.moments(mc_samples, axes=[0])
        loss_tensor = mean
        n_samples_float = tf.cast(self.n_samples, variance.dtype)
        loss_error_tensor = tf.sqrt(variance / n_samples_float)

        return MonteCarloIntegral(loss_tensor, loss_error_tensor)


class LossLowerBound:
  r"""The function ln p(X) by Monte-Carlo integral, which is the lower bound
  of the loss `LossX`.
  
  ```math
  p(X) = E_{z \sim p(z)} \left[ p(X \mid z) \right].
  ```

  The error of the Monte-Carlo integral is computed as follow.

  ```math
  \begin{equation}
    \delta \ln p(X) = \frac{ \delta p(X) }{ p(X) }.
  \end{equation}
  ```

  wherein

  ```math
  \begin{align}
    & \left( \frac{ \delta p(X) }{ p(X) } \right)^2 \\
    = & \frac{
          \text{Var}_{z \sim p(z)} \left[ p(X \mid z) \right]
        }{
          \text{E}_{z \sim p(z)}^2 \left[ p(X \mid z) \right]
        } \\
    = & \frac{
          \text{E}_{z \sim p(z)}^2 \left[ p^2(X \mid z) \right]
        }{
          \text{E}_{z \sim p(z)}^2 \left[ p(X \mid z) \right]
        } - 1.
  \end{align}
  ```

  WARNING:
    This estimation of lower bound of the fitting by the KL-divergence is
    NOT proper, because of its large variance.
    
    Indeed, as the number of samples in the Monte-Carlo integral increases,
    the variance increases, rather than decreasing as what should be expected.
    This is caused by the large variance of p(X|z), which is a multiplication
    of p(X_i|z)s where each X_i is for one pixal of the 28*28-pixal picture
    of the MNIST dataset. (Say, the multiplication of 28*28 independent
    probabilities all with the value `0.9`, i.e. 0.9**(28*28), is extremely
    tiny.)

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
      decoder_dist = self.vae.decoder_dist(latent_samples, reuse=tf.AUTO_REUSE)
      # [n_samples] + batch_shape
      decoder_log_probs = decoder_dist.log_prob(ambient)

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
