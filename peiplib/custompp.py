"""Plot and Print Customization"""

import matplotlib.pyplot as plt
import numpy as np


# --- Customizing Numpy printing ---
np.set_printoptions(precision=4, suppress=True)


# --- Customizing Matplotlib ---
plt.style.use('grayscale')
plt.rc('font', size=16, family='serif', serif=[])
plt.rc('lines', linewidth=2)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rc('figure', figsize=[8, 7], facecolor='w')
plt.rc('savefig', format='pdf', bbox='tight')
plt.rc('image', cmap='bone')
plt.rc('text', usetex=True)
plt.rc('grid', linestyle='--', color='#d3d7cf')
