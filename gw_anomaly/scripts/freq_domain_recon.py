import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import torch
from helper_functions import (
    stack_dict_into_numpy,
    stack_dict_into_numpy_segments,
    compute_fars,
    far_to_metric,
    stack_dict_into_tensor
    )
import sys
import os.path
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
    GPU_NAME
)
DEVICE = torch.device(GPU_NAME)

from quak_predict import quak_eval

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from helper_functions import mae_torch, freq_loss_torch

def main(args):

    model_paths = ["bbh.pt", "sghf.pt", "sglf.pt", "background.pt", "glitch.pt"]
    model_paths = [f"{args.model_path}/{elem}" for elem in model_paths]
    #do eval on the data

    loss_values_SNR = dict()
    loss_values = dict()
    do_recreation_plotting=True
    if do_recreation_plotting:
        #recreation plotting
        for class_label in CLASS_ORDER:
            data = np.load(f"{args.test_data_path[:-7]}{class_label}.npy")
            if class_label in ["bbh", "sglf", "sghf"]:
                loss_values_SNR[class_label] = dict()
                data_clean = np.load(f"{args.test_data_path[:-7]}{class_label}_clean.npy")
                for SNR_ind in [4]:
                    datum = data[SNR_ind]
                    dat_clean = data_clean[SNR_ind]
                    stds = np.std(datum, axis=-1)[:, :, np.newaxis]
                    datum = datum/stds
                    dat_clean = dat_clean/stds
                    datum = torch.from_numpy(datum).float().to(DEVICE)
                    evals = quak_eval(datum, model_paths, reduce_loss=False)
                    loss_values_SNR[class_label][SNR_ind] = evals['freq_loss']
                    try:
                        os.makedirs(f"{args.savedir}/SNR_{SNR_ind}_{class_label}")
                    except FileExistsError:
                        None
                    original = []
                    recreated = []
                    for class_label_ in CLASS_ORDER:
                        original.append(evals['original'][class_label_])
                        recreated.append(evals['recreated'][class_label_])
                    original = np.stack(original, axis=1)
                    recreated = np.stack(recreated, axis=1)

                    #print(original.shape, recreated.shape)
                    #assert 0


                    #work with bbh first
                    fig, axs = plt.subplots(2, figsize=(18, 9))
                    for i in range(5):
                        axs[0].plot(recreated[0, i, 0, :], label = f"rec, {CLASS_ORDER[i]}")
                        axs[1].plot(recreated[0, i, 1, :], label = f"rec, {CLASS_ORDER[i]}")

                    axs[0].plot(original[0, 0, 0, :], c="black", label = "orig")
                    axs[1].plot(original[0, 0, 1, :], c="black", label = "orig")

                    axs[0].legend()
                    axs[1].legend()

                    fig.tight_layout()
                    fig.savefig(f"{args.savedir}/reco_{class_label}.png")
                    print("SAVEPATH", f"{args.savedir}/reco.png")
                    #assert 0

                    fig, axs = plt.subplots(2,2, figsize=(18, 9))
                    for i in range(5):
                        H = recreated[0, i, 0, :]
                        L = recreated[0, i, 1, :]
                        H = np.fft.rfft(H)
                        L = np.fft.rfft(L)
                        
                        axs[0, 0].plot(np.real(H), label = f"rec, real, {CLASS_ORDER[i]}")
                        axs[0, 1].plot(np.imag(H), label = f"rec, imag, {CLASS_ORDER[i]}")
                        axs[1, 0].plot(np.real(L), label = f"rec, real, {CLASS_ORDER[i]}")
                        axs[1, 1].plot(np.imag(L), label = f"rec, imag, {CLASS_ORDER[i]}")

                    orig_H = original[0, 0, 0, :]
                    orig_H = np.fft.rfft(orig_H)

                    orig_L = original[0, 0, 1, :]
                    orig_L = np.fft.rfft(orig_L)

                    axs[0, 0].plot(np.real(orig_H), label = f"real, orig", c="black")
                    axs[0, 1].plot(np.imag(orig_H), label = f"imag, orig", c="black")
                    axs[1, 0].plot(np.real(orig_L), label = f"real, orig", c="black")
                    axs[1, 1].plot(np.imag(orig_L), label = f"imag, orig", c="black")

                    axs[0, 0].set_title("REAL COMPONENT", fontsize=20)
                    axs[0, 1].set_title("IMAG COMPONENT", fontsize=20)
                    axs[0, 0].set_ylabel("HANFORD", fontsize=20)
                    axs[1, 0].set_ylabel("LIVINGSTON", fontsize=20)
            
                    for i in range(2):
                        for j in range(2):
                            axs[i, j].legend()
                            axs[i, j].set_xscale("log")
                    fig.tight_layout()
                    fig.savefig(f"{args.savedir}/freq_reco_{class_label}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('test_data_path', help='Path of test data',
        type=str)
    parser.add_argument('model_path', help='path to the models',
        type=str)
    parser.add_argument('savedir', help='path to save the plots',
        type=str)

    args = parser.parse_args()
    main(args)