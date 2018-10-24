import numpy as np
import matplotlib.pyplot as plt


def plot_latent(ys, zs):
  plt.figure(figsize=(8, 6)) 
  plt.scatter(zs[:,0], zs[:,1], c=np.argmax(ys, axis=1))
  plt.colorbar()
  plt.grid()
  plt.show()
