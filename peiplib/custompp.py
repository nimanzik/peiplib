"""Plot and Print Customization"""

import matplotlib.pyplot as plt
import numpy as np


# --- Customizing Numpy printing ---
np.set_printoptions(precision=4, suppress=True)


# --- Customizing Matplotlib ---
plt.style.use('grayscale')
# plt.rc('font', size=14, family='serif', serif=[])
plt.rc('font', size=14)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rc('figure', figsize=[8, 7], facecolor='w')
plt.rc('savefig', format='pdf', bbox='tight')
plt.rc('image', cmap='bone')
plt.rc('text', usetex=True)
