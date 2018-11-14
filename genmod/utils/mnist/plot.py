import numpy as np
from PIL import Image


def get_image(array):
  """
  Args:
    array: Numpy array with shape `[28*28]`.

  Returns:
    A `PIL.Image` instance.
  """
  array = 255 * array
  array = array.reshape([28, 28])
  array = array.astype(np.uint8)
  return Image.fromarray(array)
