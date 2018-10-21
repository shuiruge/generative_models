import numpy as np
import tensorflow as tf
from tfutils.monte_carlo_integral import MonteCarloIntegral
try:
  import tensorflow_probability as tfp  # pylint: disable=E0401
  tfd = tfp.distributions
except:
  tfd = tf.contrib.distributions
tfb = tfd.bijectors


def get_loss_X(get_q_z_X,
               get_p_X_z,
               p_z,
               n_samples=32,
               reuse=tf.AUTO_REUSE):
  r"""Returns the the function L(X) by Monte-Carlo integral.
  
  ```math
  L(X) := E_{z ~ q(z|X)} \left[
      \ln q(z \mid X) - \ln p(z) - \ln p(X \mid z)
  \right].
  ```

  In setting the prior distribution p(z), the very effect of high-
  dimensionality of continous distribution must be taken into account.
  For instance, set Z ~ MultivariateNormalDiag, with the `scale_diag`
  ~ `tf.sqrt(1 / z_dim)`.

  Args:
    get_q_z_X: Callable with the signature:
      Args:
        X: Tensor with shape `batch_shape + [X_dim]`.
        reuse: Boolean.
      Returns:
        An instance of `tfd.Distribution`, with batch-shape `batch_shape`
        and event-shape `[z_dim]`.
    get_p_X_z: Callable with the signature:
      Args:
        z: Tensor of the shape `batch_shape + [z_dim]`.
        reuse: Boolean.
      Returns:
        An instance of `tfd.Distribution`, with batch-shape `batch_shape`
        and event-shape `[X_dim]`.
    p_z: An instance of `tfd.Distribution`, with batch-shape `[batch_size]`
      and event-shape `[z_dim]`.
    n_samples: Positive integer tensor-like object. Based on the golden-rule
      of statistics, it shall be no less than 30.
    reuse: If reuse the variables in `get_q_z_X` and `get_p_X_z`.
    
  Returns:
    Callable with the signature:
      Args:
        X: Tensor of the shape `batch_shape + [X_dim]`.
    Returns:
      An instance of `MonteCarloIntegral` with shape `batch_shape`.
  """

  def loss_X(X, name='loss_X'):
    """Returns the tensors for L(X) and its error.

    Args:
      X: Tensor of the shape `batch_shape + [X_dim]`.

    Returns:
      An instance of `MonteCarloIntegral` with shape `batch_shape`.
    """
    with tf.name_scope(name):
      # Get the distribution q(z|X)
      q_z_X = get_q_z_X(X, reuse=reuse)
      
      # Get the distribution p(X|z)
      # [n_samples] + batch_shape + [z_dim]
      z_samples = q_z_X.sample(n_samples)
      p_X_z = get_p_X_z(z_samples, reuse=reuse)

      # log_q(z|X) - log_p(z) - log_p(X|z)
      # [n_samples] + batch_shape
      mc_samples = (q_z_X.log_prob(z_samples) -
                    p_z.log_prob(z_samples) -
                    p_X_z.log_prob(X))
      # E{z ~ q(z|X)} [...]
      mean, variance = tf.nn.moments(mc_samples, axes=[0])
      loss_X_tensor = mean
      n_samples_float = tf.cast(n_samples, variance.dtype)
      loss_X_error_tensor = tf.sqrt(variance / n_samples_float)
      return MonteCarloIntegral(loss_X_tensor, loss_X_error_tensor)

  return loss_X


def get_log_p_X(get_p_X_z,
                p_z,
                n_samples=32,
                epsilon=1e-8,
                reuse=None):
  r"""Returns the function ln p(X) by Monte-Carlo integral.
  
  ```math
  p(X) = E_{z \sim p(z)} \left[ p(X \mid z) \right].
  ```

  XXX: Some bugs here in the computation of the error of the Monte-Carlo
    integral.

  Args:
    get_p_X_z: Callable with the signature:
      Args:
        z: Tensor of the shape `batch_shape + [z_dim]`.
        reuse: Boolean.
      Returns:
        An instance of `tfd.Distribution`, with batch-shape `batch_shape`
        and event-shape `[X_dim]`.
    p_z: An instance of `tfd.Distribution`, with batch-shape `batch_size`
      and event-shape `z_dim`.
    n_samples: Positive integer.
    reuse: If reuse the variables in `get_p_X_z`.
    
  Returns:
    Callable with the signature:
      Args:
        X: Tensor of the shape `batch_shape + [X_dim]`.
        name: String.
      Returns:
        An instance of `MonteCarloIntegral` with shape `batch_shape`.
  """
  log_n_samples = tf.log(tf.cast(n_samples, p_z.dtype),
                         name='log_n_samples')

  def log_expectation(log_samples, name='log_expectation'):
    """ln E[ exp(log_samples) ]
    
    Args:
      log_samples: Tensor of the shape `[n_samples]` + batch-shape +
        event-shape.
      
    Returns:
      Tensor of the shape batch_shape + event-shape.
    """
    with tf.name_scope(name):
      return tf.reduce_logsumexp(log_samples - log_n_samples, axis=0)

  def log_variance(log_samples, name='log_variance'):
    """ln Var[ exp(log_samples) ]
    
    Args:
      log_samples: Tensor of the shape `[n_samples]` + batch-shape +
        event-shape.
      
    Returns:
      Tensor of the shape batch_shape + event-shape.
    """
    with tf.name_scope(name):
      log_mean = log_expectation(log_samples)
      # Make the shape the same as `log_mc_samples`
      log_mean = tf.ones_like(log_samples) * tf.expand_dims(log_mean, axis=0)

      # delta[i] := exp(log_samples[i]) - exp(log_mean)
      # ln{ delta[i]**2 }
      # The same shape as `log_samples`
      log_delta_square = tf.add(
          2.0 * log_samples,
          tf.log(tf.square(1.0 - tf.exp(log_mean - log_samples))),
          name='log_delta_square')

      # In some cases, simplication is employed
      # - When log_samples[i] >> log_mean, approximated by 2 * log_mean
      # - When log_samples[i] << log_mean, approximated by 2 * log_samples[i]
      log_delta_square = tf.where(
          log_samples - log_mean < tf.log(epsilon),
          2.0 * log_mean,
          tf.where(log_samples - log_mean > -tf.log(epsilon),
                  2.0 * log_samples,
                  log_delta_square),
          name='simplified_log_delta_square')

      log_variance_tensor = log_expectation(log_delta_square,
                                            name='log_variance_tensor')
      return log_variance_tensor

  def log_p_X(X, name='log_p_X'):
    """Returns the tensor of ln p(X). This serves as the lower bound of
    the loss by KL-divergence, evaluating the fitting.

    Args:
      X: Tensor of the shape `batch_shape + [X_dim]`.
      name: String.
      
    Returns:
      An instance of `MonteCarloIntegral` with shape `batch_shape`.
    """
    with tf.name_scope(name):      
      # [n_samples] + batch_shape + [z_dim]
      z_samples = p_z.sample(n_samples)
      p_X_z = get_p_X_z(z_samples, reuse=reuse)
      # [n_samples] + batch_shape
      log_p_X_z_samples = p_X_z.log_prob(X)

      # E_{z~p(z)} [ p(X|z) ]
      # batch_shape
      log_p_X_tensor = log_expectation(log_p_X_z_samples,
                                       name='log_p_X_tensor')

      # Error of Monte-Carlo integral
      log_variance_tensor = log_variance(log_p_X_z_samples,
                                         name='log_variance_tensor')
      log_error_tensor = tf.subtract(0.5 * log_variance_tensor,
                                     log_p_X_tensor,
                                     name='log_error_tensor')
      error_tensor = tf.exp(log_error_tensor,
                            name='error_tensor')
      return MonteCarloIntegral(log_p_X_tensor, error_tensor)

  return log_p_X
