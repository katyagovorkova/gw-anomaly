import os
import sys
import argparse
import numpy as np
import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import SMOOTHING_KERNEL_SIZES

hists = snakemake.params[0]
save_path = snakemake.params[1]

for kernel_len in SMOOTHING_KERNEL_SIZES:
    new_hist = np.zeros((np.load(f"{hists[0][:-4]}_{kernel_len}.npy").shape))
    for hist in hists:
        mod_path = f"{hist[:-4]}_{kernel_len}.npy"
        past_hist = np.load(mod_path)
        new_hist += past_hist

    np.save(f"{save_path[:-4]}_{kernel_len}.npy", new_hist)
    
    #to conform with the old, no-smoothing variant, so there aren't more errors in the plotting code
    if kernel_len == 1:
        np.save(save_path, new_hist)