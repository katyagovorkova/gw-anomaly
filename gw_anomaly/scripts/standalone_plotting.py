import time
import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl
import scipy.stats as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from labellines import labelLines
from helper_functions import (
    stack_dict_into_numpy,
    stack_dict_into_numpy_segments,
    compute_fars,
    far_to_metric
)
from models import LinearModel

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SEG_NUM_TIMESTEPS,
    SAMPLE_RATE,
    CLASS_ORDER,
    SPEED,
    NUM_IFOS,
    IFO_LABELS,
    RECREATION_WIDTH,
    RECREATION_HEIGHT_PER_SAMPLE,
    RECREATION_SAMPLES_PER_PLOT,
    SNR_VS_FAR_BAR,
    SNR_VS_FAR_HORIZONTAL_LINES,
    SNR_VS_FAR_HL_LABELS,
    SEGMENT_OVERLAP,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN,
    VARYING_SNR_LOW,
    VARYING_SNR_HIGH,
    GPU_NAME,
    RETURN_INDIV_LOSSES,
    CURRICULUM_SNRS,
    FACTORS_NOT_USED_FOR_FM,
    HRSS_VS_FAR_BAR,
    DO_SMOOTHING,
    SMOOTHING_KERNEL,
    SMOOTHING_KERNEL_SIZES,
    DATA_LOCATION)

DEVICE = torch.device(GPU_NAME)

def heuristic_cut_effic_plot(snrs_dict, heuristics_dict, savedir):
    # make all the plots at once
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    tags = ['bbh', 'wnbhf', 'supernova', 'wnblf', 'sglf', 'sghf']
    
    for i in range(6):
        row, col = i//3, i % 3

        snrs = snrs_dict[tags[i]]
        heur = heuristics_dict[tags[i]]

        # need to bin by SNR
        _, snr_bins = np.histogram(snrs, bins=45, range=(5, 50))
        heuristic_fraction = np.zeros(snr_bins.shape)
        heuristic_counts = np.zeros(snr_bins.shape)

        for j in range(len(snrs)):
            placement = np.searchsorted(snr_bins, snrs[j])
            heuristic_fraction[placement] += heur[j]
            heuristic_counts[placement] += 1
        # "law of large numbers"...hoping there's no zeros
        heuristic_fraction = 100 - (heuristic_fraction / heuristic_counts * 100)
        #filter
        axs[row, col].plot(snr_bins, heuristic_fraction)

        axs[row, col].set_title(tags[i])
        if row == 1:
            axs[row, col].set_xlabel("SNR")

        if col == 0:
            axs[row, col].set_ylabel("% of signal lost")

    fig.savefig(savedir + "heuristic_cut_effic.pdf", dpi=300)


def main(args):
    try:
        os.makedirs(args.plot_savedir)
    except FileExistsError:
        None

    do_heuristic_efficiency = True

    if do_heuristic_efficiency:
        fm_model_path = ("/home/katya.govorkova/gwak-paper-final-models/trained/fm_model.pt")
        fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
        fm_model.load_state_dict(torch.load(
            fm_model_path, map_location=GPU_NAME))
        norm_factors = np.load(f"/home/katya.govorkova/gwak-paper-final-models/trained/norm_factor_params.npy")
        linear_weights = fm_model.layer.weight.detach()#.cpu().numpy()
        bias_value = fm_model.layer.bias.detach()#.cpu().numpy()
        linear_weights[:, -2] += linear_weights[:, -1]
        linear_weights = linear_weights[:, :-1]
        norm_factors = norm_factors[:, :-1]

        mean_norm = torch.from_numpy(norm_factors[0]).to(DEVICE)#[:-1]
        std_norm = torch.from_numpy(norm_factors[1]).to(DEVICE)#[:-1]

        tags = ['bbh', 'sghf', 'wnbhf', 'supernova', 'wnblf', 'sglf', 'sghf']
        data_dict = {}
        snrs_dict = {}
        heuristics_dict = {}
        for tag in tags:

            print(f'loading {tag}')
            ts = time.time()
            data = np.load(f'{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy')
            #data = np.delete(data, FACTORS_NOT_USED_FOR_FM, -1)
            data = torch.from_numpy(data).to(DEVICE).float()

            print(f'{tag} loaded in {time.time()-ts:.3f} seconds')

            data = (data - mean_norm) / std_norm
            data = data#[1000:]

            snrs = np.load(f'{DATA_LOCATION}/{tag}_varying_snr_SNR.npz.npy')#[1000:]
            passed_heuristics = np.load(f'{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals_heuristic_res.npy')
            # hrss = np.load(f'/home/katya.govorkova/gwak-paper-final-models/data/{tag}_varying_snr_hrss.npz.npy')

            data_dict[tag] = data
            snrs_dict[tag] = snrs
            heuristics_dict[tag] = passed_heuristics

        heuristic_cut_effic_plot(snrs_dict, heuristics_dict, args.plot_savedir)
            



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_predicted_path', help='Path to data directory',
                        type=str)

    parser.add_argument('plot_savedir', help='Required output directory for saving plots',
                        type=str)

    args = parser.parse_args()
    main(args)
