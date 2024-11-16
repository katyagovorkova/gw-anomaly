import os
import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import FACTORS_NOT_USED_FOR_FM

# We need to remove the values not used for the final metric
# we keep the Pearson to be able to compare NPLM to the methods paper

# we add normalization to match our results
means, stds = np.load(f'/home/katya.govorkova/gwak-paper-final-models/trained/norm_factor_params.npy')

paths = [f'output/O3av2/evaluated/{signal}_varying_snr_evals.npy' for signal \
    in ['sglf', 'sghf', 'bbh', 'supernova', 'wnblf', 'wnbhf']]


for path in paths:

    data = np.load(path).copy()
    data = np.delete(data, FACTORS_NOT_USED_FOR_FM, -1)
    data = (data - means) / stds

    new_path = path.replace('_evals.npy', '_evals_16.npy')

    np.save(new_path, data)
    print(f'Saved data in {new_path} with a shape {data.shape}')

for i in range(16):
    plt.plot(data[:,i].flatten())

plt.show()