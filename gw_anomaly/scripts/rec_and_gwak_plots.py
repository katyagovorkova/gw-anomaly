import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import torch

from gwak_predict import quak_eval as gwak_eval
from models import LinearModel
from helper_functions import stack_dict_into_tensor

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SEG_NUM_TIMESTEPS,
    SAMPLE_RATE,
    CLASS_ORDER,
    NUM_IFOS,
    IFO_LABELS,
    RECREATION_WIDTH,
    RECREATION_HEIGHT_PER_SAMPLE,
    RECREATION_SAMPLES_PER_PLOT,
    GPU_NAME,
    FACTORS_NOT_USED_FOR_FM,
    MODELS_LOCATION
    )
DEVICE = torch.device(GPU_NAME)


def density_plot(x, y):
    # Define the borders
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY
    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    return xx, yy, f


def corner_plotting(
        data: list[np.ndarray],
        labels: list[str],
        plot_savedir: str,
        enforce_lim: bool=True,
        contour: bool=True,
        loglog: bool=False,
        do_cph: bool=False,
        save_1d_hist: bool=False,
        SNR_ind: int=None):

    # corner plot, BIL, LAL
    N = len(labels)
    fig, axs = plt.subplots(N, N, figsize=(20, 20))
    oneD_hist_kwargs = dict(histtype='stepfilled',
                            alpha=1, density=True, bins=40)
    # hide all of the ones not used
    for i in range(N):
        for j in range(i + 1, N):
            axs[i, j].axis('off')

    cmaps = [
        'Purples',
        'Blues',
        'Greens',
        'Reds',
        'Oranges']

    one_D_colors = [
        'purple',
        'steelblue',
        'darkgreen',
        'salmon',
        'goldenrod'
    ]

    print('123')
    print(labels)

    # do the 1-d plots
    for i in range(N):
        norm_factor = 0
        for j, class_data in enumerate(data):

            if labels[j] == 'glitches':
                LBL = 'Glitches'
            elif labels[j] == 'sglf':
                LBL = 'SG Injection, 64-512 Hz'
            elif labels[j] == 'sghf':
                LBL = 'SG Injection, 512-1024 Hz'
            elif labels[j] == 'bbh':
                LBL = 'BBH'
            elif labels[j] == 'background':
                LBL = 'Background'
            else:
                LBL = labels[j]

            axs[i, i].hist(class_data[:, i], color=one_D_colors[j], **oneD_hist_kwargs, label=LBL)
            if save_1d_hist:
                np.save(f'{plot_savedir}/one_d_hist_{i}_{j}.npy', class_data[:, i])
            if enforce_lim:
                axs[i, i].set_xlim(0, 1.2)

        axs[i, i].legend()

    log_scaling = False

    corner_plot_hist = [labels]
    # do 2-d plots
    for i in range(N):
        for j in range(i):
            for k, class_data in enumerate(data):

                A, B = class_data[:, i], class_data[:, j]

                if contour:
                    xx, yy, f = density_plot(A, B)
                    cset = axs[i, j].contour(yy, xx, f, cmap=cmaps[k])
                    axs[i, j].clabel(cset, inline=1, fontsize=10)
                    if enforce_lim and not loglog:
                        axs[i, j].set_xlim(0, 1.2)
                        axs[i, j].set_ylim(0, 1.2)
                    if loglog:
                        axs[i, j].loglog()
                    # save these values somehow
                    corner_plot_hist.append([i, j, k, yy, xx, f])

                else:
                    axs[i, j].scatter(B, A, s=15, c=cmaps[k][:-1])
                    if enforce_lim:
                        axs[i, j].set_xlim(0, 1.2)
                        axs[i, j].set_ylim(0, 1.2)
                    if loglog:
                        axs[i, j].loglog()

    # axis labels
    for i in range(N):
        if labels[i] == 'glitches':
            lbl = 'Glitches'
        elif labels[i] == 'sglf':
            lbl = 'SG Injection, 64-512 Hz'
        elif labels[i] == 'sghf':
            lbl = 'SG Injection, 512-1024 Hz'
        elif labels[i] == 'bbh':
            lbl = 'BBH'
        elif labels[i] == 'background':
            lbl = 'Background'
        else:
            lbl = labels[i]
        axs[i, 0].set_ylabel(lbl, fontsize=15)
        axs[-1, i].set_xlabel(lbl, fontsize=15)
    if not loglog:
        fig.tight_layout()
        fig.savefig(plot_savedir + f'/gwak_plot.pdf')
    else:
        fig.tight_layout()
        fig.savefig(plot_savedir + f'/gwak_plot_freq.pdf')

    # save the corner plot hist
    corner_plot_hist = np.array(corner_plot_hist, dtype='object')
    if do_cph:
        np.save(f'{plot_savedir}/cph.npy', corner_plot_hist)


def recreation_plotting(
        data_original,
        data_recreated,
        data_cleaned,
        savedir,
        class_name):

    ts = np.linspace(0, 1000 * SEG_NUM_TIMESTEPS /
                     SAMPLE_RATE, SEG_NUM_TIMESTEPS)
    colors = [
        'purple',
        'steelblue',
        'darkgreen',
        'salmon',
        'goldenrod'
    ]
    i = CLASS_ORDER.index(class_name)
    try:
        os.makedirs(f'{savedir}/')
    except FileExistsError:
        None
    orig_samps = data_original[:RECREATION_SAMPLES_PER_PLOT, i, :, :]
    recreated_samps = data_recreated[:RECREATION_SAMPLES_PER_PLOT, :, :, :]

    # make the plot showing only original, recreated for that class

    if RECREATION_SAMPLES_PER_PLOT > 1:

        fig, axs = plt.subplots(RECREATION_SAMPLES_PER_PLOT, 2, figsize=(
            RECREATION_WIDTH, RECREATION_SAMPLES_PER_PLOT * RECREATION_HEIGHT_PER_SAMPLE))
        if data_cleaned is not None:
            axs[j, k].plot(ts, data_cleaned[j, k, :],
                           label='Signal', c='pink')

        for j in range(RECREATION_SAMPLES_PER_PLOT):
            for k in range(NUM_IFOS):
                mae = np.mean(
                    np.abs(orig_samps[j, k, :] - recreated_samps[j, i, k, :]))
                axs[j, k].plot(ts, orig_samps[j, k, :],
                               label='Signal + Noise', c='black')
                axs[j, k].plot(ts, recreated_samps[j, i, k, :],
                    label=f'{class_name}, mae:{mae:.2f}', c=colors[i])

                axs[j, k].grid()
                axs[j, k].set_title(IFO_LABELS[k])
                axs[j, k].legend()
                if k == 0:
                    axs[j, k].set_ylabel(r'Whitened Strain, $\sigma = 1$')
                axs[j, k].set_xlabel('Time (ms)')

            plt.tight_layout()
            fig.savefig(f'{savedir}/one_to_one.pdf', dpi=300)
            plt.clf()
    # make the plot showing original, recreated for all classes

    rename_map = {
        'background': 'Background',
        'bbh': 'BBH',
        'glitches': 'Glitches',
        'sglf': 'SG 64-512 Hz',
        'sghf': 'SG 512-1024 Hz'
    }
    if RECREATION_SAMPLES_PER_PLOT > 1:
        fig, axs = plt.subplots(RECREATION_SAMPLES_PER_PLOT, 2, figsize=(
            RECREATION_WIDTH, RECREATION_SAMPLES_PER_PLOT * RECREATION_HEIGHT_PER_SAMPLE))

        for j in range(RECREATION_SAMPLES_PER_PLOT):
            for k in range(NUM_IFOS):

                axs[j, k].grid()
                axs[j, k].set_title(IFO_LABELS[k])
                axs[j, k].legend()
                if k == 0:
                    axs[j, k].set_ylabel(r'Whitened Strain, $\sigma = 1$')
                axs[j, k].set_xlabel('Time (ms)')

                axs[j, k].plot(ts, orig_samps[j, k, :],
                               label='Signal + Noise', c='black', alpha=0.55)

                for l_name, l_color in zip(['bbh', 'sglf', 'sghf'],
                        ['steelblue', 'salmon', 'goldenrod']):
                    l = CLASS_ORDER.index(l_name)
                    mae = np.mean(
                        np.abs(orig_samps[j, k, :] - recreated_samps[j, l, k, :]))
                    alpha = 1
                    if l_name != class_name:
                        alpha = 0.5
                    axs[j, k].plot(ts, recreated_samps[j, l, k, :],
                        label=f'{rename_map[l_name]}, mae: {mae:.2f}',
                        c=l_color)

                    if data_cleaned is not None:
                        axs[j, k].plot(ts, data_cleaned[j, k, :],
                                       label='Signal', c='pink', alpha=0.8)

        plt.tight_layout()
        fig.savefig(f'{savedir}/recreation_{class_name}_sig.pdf', dpi=300)

        fig, axs = plt.subplots(RECREATION_SAMPLES_PER_PLOT, 2, figsize=(
            RECREATION_WIDTH, RECREATION_SAMPLES_PER_PLOT * RECREATION_HEIGHT_PER_SAMPLE))
        for j in range(RECREATION_SAMPLES_PER_PLOT):
            for k in range(NUM_IFOS):
                axs[j, k].grid()
                axs[j, k].set_title(IFO_LABELS[k])
                axs[j, k].legend()
                if k == 0:
                    axs[j, k].set_ylabel(r'Whitened Strain, $\sigma = 1$')
                axs[j, k].set_xlabel('Time (ms)')

                axs[j, k].plot(ts, orig_samps[j, k, :],
                               label='Signal + Noise', c='black', alpha=0.55)

                for l_name, l_color in zip(['background', 'glitches'],
                    ['purple', 'darkgreen']):
                    l = CLASS_ORDER.index(l_name)
                    mae = np.mean(
                        np.abs(orig_samps[j, k, :] - recreated_samps[j, l, k, :]))
                    alpha = 1
                    if l_name != class_name:
                        alpha = 0.5
                    axs[j, k].plot(ts, recreated_samps[j, l, k, :],
                        label=f'{rename_map[l_name]}, mae: {mae:.2f}',
                        c=l_color)

                    if data_cleaned is not None:
                        axs[j, k].plot(ts, data_cleaned[j, k, :],
                                       label='Signal', c='pink', alpha=0.8)

        plt.tight_layout()
        fig.savefig(f'{savedir}/recreation_{class_name}_bkg.pdf', dpi=300)

    else:
        j = 0
        fig, axs = plt.subplots(RECREATION_SAMPLES_PER_PLOT, 2, figsize=(
            RECREATION_WIDTH, RECREATION_SAMPLES_PER_PLOT * RECREATION_HEIGHT_PER_SAMPLE))
        for k in range(NUM_IFOS):

            if data_cleaned is not None:
                axs[k].plot(ts, orig_samps[
                            j, k, :], label='Signal + Noise, AE input', c='black',
                            alpha=0.55)
            else:
                # for glitch, bkg samples
                axs[k].plot(ts, orig_samps[j, k, :],
                            label='Signal + Noise, AE input', c='black', alpha=0.55)
            if data_cleaned is not None:
                axs[k].plot(ts, data_cleaned[j, k, :],
                            label='Signal', c='pink')

            for l_name, l_color in zip(['bbh', 'sglf', 'sghf'],
                        ['steelblue', 'salmon', 'goldenrod']):
                l = CLASS_ORDER.index(l_name)
                mae = np.mean(
                    np.abs(orig_samps[j, k, :] - recreated_samps[j, l, k, :]))
                alpha = 1
                linewidth = 2.1
                if l_name != class_name:
                    alpha = 0.75
                    linewidth = 1.45
                axs[k].plot(ts, recreated_samps[j, l, k, :], label=f'{rename_map[l_name]}, mae: {mae:.2f}',
                    c=l_color)

                axs[k].grid()
                axs[k].set_title(IFO_LABELS[k], fontsize=20)
                axs[k].legend(loc='upper left')
                if k == 0:
                    axs[k].set_ylabel(
                        r'Whitened Strain, $\sigma = 1$', fontsize=20)
                axs[k].grid()
                axs[k].set_xlabel('Time (ms)', fontsize=20)

        plt.tight_layout()
        fig.savefig(f'{savedir}/recreation_{class_name}_sig.pdf', dpi=300)

        fig, axs = plt.subplots(RECREATION_SAMPLES_PER_PLOT, 2, figsize=(
                RECREATION_WIDTH, RECREATION_SAMPLES_PER_PLOT * RECREATION_HEIGHT_PER_SAMPLE))

        for k in range(NUM_IFOS):

            if data_cleaned is not None:
                axs[k].plot(ts, orig_samps[
                            j, k, :], label='Signal + Noise, AE input',
                            c='black', alpha=0.55)
            else:
                # for glitch, bkg samples
                axs[k].plot(ts, orig_samps[j, k, :],
                            label='Signal + Noise, AE input', c='black', alpha=0.55,)
            if data_cleaned is not None:
                axs[k].plot(ts, data_cleaned[j, k, :],
                            label='Signal', c='pink')

            for l_name, l_color in zip(['background', 'glitches'],
                    ['purple', 'darkgreen']):
                l = CLASS_ORDER.index(l_name)

                mae = np.mean(
                    np.abs(orig_samps[j, k, :] - recreated_samps[j, l, k, :]))
                alpha = 1
                linewidth = 2.1
                if l_name != class_name:
                    alpha = 0.75
                    linewidth = 1.45
                axs[k].plot(ts, recreated_samps[j, l, k, :], label=f'{rename_map[l_name]}, mae: {mae:.2f}',
                    c=l_color)

                axs[k].grid()
                axs[k].set_title(IFO_LABELS[k], fontsize=20)
                axs[k].legend(loc='upper left')
                if k == 0:
                    axs[k].set_ylabel(
                        r'Whitened Strain, $\sigma = 1$', fontsize=20)
                axs[k].grid()
                axs[k].set_xlabel('Time (ms)', fontsize=20)

        plt.tight_layout()
        fig.savefig(f'{savedir}/recreation_{class_name}_bkg.pdf', dpi=300)


def main(args):

    model_paths = args.model_path if not args.from_saved_models else \
        [os.path.join(MODELS_LOCATION, os.path.basename(f)) for f in args.model_path]

    fm_model_path = args.fm_model_path  # if not args.from_saved_fm_model else \

    model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
    model.load_state_dict(torch.load(
        fm_model_path, map_location=GPU_NAME))
    weight = (model.layer.weight.data.cpu().numpy()[0])
    weights = []
    for i in range(5):
        arr = np.zeros(weight.shape)
        arr[2 * i] = weight[2 * i]
        arr[2 * i + 1] = weight[2 * i + 1]
        arr[2 * i + 3] = weight[2 * i + 3]
        weights.append(arr)  # cut out pearson

    loss_values_SNR = dict()
    loss_values = dict()
    do_recreation_plotting = True

    # recreation plotting
    for class_label in CLASS_ORDER:
        if class_label in ['bbh', 'sglf', 'sghf']:
            loss_values_SNR[class_label] = dict()
            data = np.load(f'{args.test_data_path[:-7]}{class_label}.npz')['noisy']
            data_clean = np.load(f'{args.test_data_path[:-7]}{class_label}.npz')['clean']

            SNR_ind = 4
            datum = data[SNR_ind]
            dat_clean = data_clean[SNR_ind]
            stds = np.std(datum, axis=-1)[:, :, np.newaxis]
            datum = datum / stds
            dat_clean = dat_clean / stds
            datum = torch.from_numpy(datum).float().to(DEVICE)
            evals = gwak_eval(datum, model_paths, device=DEVICE, reduce_loss=False)
            loss_values_SNR[class_label][SNR_ind] = evals['freq_loss']
            original = []
            recreated = []
            for class_label_ in CLASS_ORDER:
                original.append(evals['original'][class_label_])
                recreated.append(evals['recreated'][class_label_])
            original = np.stack(original, axis=1)
            recreated = np.stack(recreated, axis=1)
            recreation_plotting(original,
                                recreated,
                                dat_clean,
                                f'{args.savedir}/{class_label}/',
                                class_label)
        else:
            data = np.load(f'{args.test_data_path[:-7]}{class_label}.npz')['data']
            datum = data
            stds = np.std(datum, axis=-1)[:, :, np.newaxis]
            datum = datum / stds
            datum = torch.from_numpy(datum).float().to(DEVICE)
            evals = gwak_eval(datum, model_paths, device=DEVICE, reduce_loss=False)
            loss_values[class_label] = evals['freq_loss']
            try:
                os.makedirs(f'{args.savedir}/{class_label}/')
            except FileExistsError:
                None
            original = []
            recreated = []
            for class_label_ in CLASS_ORDER:
                original.append(evals['original'][class_label_])
                recreated.append(evals['recreated'][class_label_])
            original = np.stack(original, axis=1)
            recreated = np.stack(recreated, axis=1)
            recreation_plotting(original,
                                recreated,
                                None,
                                f'{args.savedir}/{class_label}/',
                                class_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('test_data_path', help='Path of test data',
                        type=str)
    parser.add_argument('model_path', help='path to the models',
                        nargs='+', type=str)
    parser.add_argument('from_saved_models', type=bool,
                        help='If true, use the pre-trained models from MODELS_LOCATION in config, otherwise use models trained with the pipeline.')
    parser.add_argument('fm_model_path', help='path to the final metric model',
                        type=str)
    parser.add_argument('from_saved_fm_model', type=bool,
                        help='If true, use the pre-trained models from MODELS_LOCATION in config, otherwise use models trained with the pipeline.')
    parser.add_argument('savedir', help='path to save the plots',
                        type=str)

    args = parser.parse_args()
    main(args)
