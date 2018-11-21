import abc
import tensorflow as tf
from genmod.f_gan.base import BaseFGAN
from genmod.vae.base import BaseVAE


class GanVae(BaseVAE, BaseFGAN, metaclass=abc.ABCMeta):

  def __init__(self, gan, vae):
    assert isinstance(gan, BaseFGAN)
    assert isinstance(vae, BaseVAE)
    assert are_same_distribution(gan.prior, vae.prior)

    self.gan = gan
    self.vae = vae

  def discriminator(self, ambient, reuse):
    return self.gan.discriminator(ambient, reuse)

  def generator(self, latent, reuse):
    return self.gan.generator(latent, reuse)

  @property
  def prior(self):
    return self.gan.prior

  def encoder(self, ambient, reuse):
    return self.vae.encoder(ambient, reuse)

  @abc.abstractmethod
  def decoder_dist(self, dist_parameter):
    r"""Maps from the parameter-space of the decoder-distribution to
    the distribution. For instance, if the decoder-distribution is to
    be `Bernoulli`, and the `self.generator` returns the logits, then

    ```python
    def decoder_dist(self, logits):
      return Bernoulli(logits=logits)
    ```

    TODO: Needs some checking.

    Args:
      dist_parameter: Tensor.

    Returns:
      A `Distribution` instance.
    """
    pass

  def decoder(self, latent, reuse):
    return self.decoder_dist(self.generator(latent, reuse))

  def gan_loss(self, data, reuse=tf.AUTO_REUSE):
    return self.gan.loss(data, reuse=reuse)

  def vae_loss(self, data, reuse=tf.AUTO_REUSE):
    # The MRO of inheriting ensures that the `self.loss` calls for that of
    # `BaseVAE`, but with its method `self.decoder` overrided, constructed
    # from `self.generator` of the GAN.
    return self.loss(data, reuse=reuse)


def are_same_distribution(distribution, other_distribution):
  """
  Args:
    distribution: A distribution.
    other_distribution: A distribution.

  Returns:
    Boolean.
  """
  if distribution.__class__ != other_distribution.__class__:
    return False
  if distribution.event_shape != other_distribution.event_shape:
    return False
  if distribution.batch_shape != other_distribution.batch_shape:
    return False
  return True
