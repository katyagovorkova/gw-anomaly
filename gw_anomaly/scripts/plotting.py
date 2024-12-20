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
    DO_SMOOTHING,
    SMOOTHING_KERNEL,
    SMOOTHING_KERNEL_SIZES,
    VERSION)

DEVICE = torch.device(GPU_NAME)


def calculate_means(metric_vals, snrs, bar):
    # helper function for SNR vs FAR plot
    means, stds = [], []
    snr_plot = []

    for i in np.arange(VARYING_SNR_LOW, VARYING_SNR_HIGH, bar):

        points = []
        for shift in range(bar):
            for elem in np.where(((snrs - shift).astype(int)) == i)[0]:
                points.append(elem)
        if len(points) == 0:
            continue

        snr_plot.append(i + bar / 2)
        MV = []
        for point in points:
            MV.append(metric_vals[point])
        MV = np.array(MV)
        means.append(np.mean(MV))
        stds.append(np.std(MV))

    return snr_plot, means, stds


def amp_measure_vs_far_plotting(
        datas,
        amp_measures, #rename from snr to encompass both snr and hrss
        metric_coefs,
        far_hist,
        tags,
        savedir,
        special,
        bias,
        hrss=False):
    fig, axs = plt.subplots(1, figsize=(12, 8))
    colors = {
        'bbh': 'steelblue',
        'sglf': 'salmon',
        'sghf': 'goldenrod',
        'wnbhf': 'purple',
        'wnblf': 'hotpink',
        'supernova': 'darkorange'
    }

    if not hrss:
        axs.set_xlabel(f'SNR', fontsize=20)
    else:
        axs.set_xlabel(f'hrss', fontsize=20)

    axs.set_ylabel('Final metric value, a.u.', fontsize=20)
    populate = dict()
    for k in range(len(datas)):
        data = datas[k]
        amp_measure = amp_measures[k]
        tag = tags[k]

        if RETURN_INDIV_LOSSES:
            fm_vals = metric_coefs(torch.from_numpy(
                data).float().to(DEVICE)).detach().cpu().numpy()
        else:
            fm_vals = np.dot(data, metric_coefs)

        fm_vals = np.apply_along_axis(lambda m: np.convolve(m, np.ones(5)/5, mode='same'),
            axis=1,
            arr=fm_vals)
        fm_vals = np.min(fm_vals, axis=1)
        populate[tag] = fm_vals
        amp_measure_plot, means_plot, stds_plot = calculate_means(
            fm_vals, amp_measure, bar=SNR_VS_FAR_BAR)

        means_plot, stds_plot = np.array(means_plot), np.array(stds_plot)
        rename_map = {
            'background': 'Background',
            'bbh': 'BBH',
            'glitches': 'Glitch',
            'sglf': 'SG 64-512 Hz',
            'sghf': 'SG 512-1024 Hz',
            'wnblf': 'WNB 40-400 Hz',
            'wnbhf': 'WNB 400-1000 Hz',
            'supernova': 'Supernova'
        }
        tag_ = rename_map[tag]

        axs.plot(amp_measure_plot, means_plot, color=colors[tag], label=f'{tag_}', linewidth=2)
        axs.fill_between(amp_measure_plot,
                         (means_plot) - stds_plot / 2,
                         (means_plot) + stds_plot / 2,
                         alpha=0.15,
                         color=colors[tag])

    for i, label in enumerate(SNR_VS_FAR_HL_LABELS):
        metric_val_label = far_to_metric(
            SNR_VS_FAR_HORIZONTAL_LINES[i], far_hist)
        if metric_val_label is not None:
            axs.axhline(y=metric_val_label, alpha=0.8**i, label=f'1/{label}', c='black')

    labelLines(axs.get_lines(), zorder=2.5, xvals=(
        15, 20, 15, 30, 35, 40, 25, 30, 35, 40, 45))
    axs.set_title(special, fontsize=20)
    if not hrss:
        axs.set_xlim(VARYING_SNR_LOW + SNR_VS_FAR_BAR / 2,
                     VARYING_SNR_HIGH - SNR_VS_FAR_BAR / 2)
    else:
        None # figure this out
    axs.set_ylim(-40,0)

    plt.grid(True)
    fig.tight_layout()
    plt.savefig(f'{savedir}/{special}.pdf', dpi=300)
    plt.close()

    return populate


def fake_roc_plotting(far_hist, savedir):
    datapoint_to_seconds = SEGMENT_OVERLAP / SAMPLE_RATE
    total_datapoints = far_hist.sum()
    total_seconds = total_datapoints * datapoint_to_seconds
    x_plot = []
    y_plot = []
    for i in range(len(far_hist)):
        total_below = np.sum(far_hist[:i])
        x_plot.append(i * HISTOGRAM_BIN_DIVISION - HISTOGRAM_BIN_MIN)
        y_plot.append(total_below / total_seconds)

    plt.figure()
    plt.plot(x_plot, y_plot)
    plt.yscale('log')
    plt.xlabel('Metric value')
    plt.ylabel('Corresponding FAR, Hz')
    plt.xlim(-50, 50)

    plt.savefig(f'{savedir}/fake_roc.pdf', dpi=300)


def three_panel_plotting(
    strain,
    data,
    snr,
    metric_coefs,
    far_hist,
    tag,
    plot_savedir,
    bias,
    weights):
    # doing only one sample, for now
    print('Warning: three panel plot has incorrect x-axis, implement this!')
    fig, axs = plt.subplots(3, figsize=(8, 14))

    colors = [
        'purple',
        'steelblue',
        'darkgreen',
        'salmon',
        'goldenrod',
        'sienna',
        'black'
    ]
    labels = [
        'Background',
        'BBH',
        'Glitch',
        'SG 64-512 Hz',
        'SG 512-1024 Hz',
        'Freq domain corr.',
        'Pearson'
    ]

    ifo_colors = {
        0: 'goldenrod',
        1: 'darkgreen'}

    if RETURN_INDIV_LOSSES:
        fm_vals = metric_coefs(torch.from_numpy(
            data).float().to(DEVICE)).detach().cpu().numpy() - bias
    else:
        fm_vals = np.dot(data, metric_coefs)

    print(f'shape before convolution {fm_vals.shape}')
    fm_vals = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(SMOOTHING_KERNEL)/SMOOTHING_KERNEL, mode='same'),
        axis=0,
        arr=fm_vals)
    print(f'shape after convolution before min {fm_vals.shape}')
    # fm_vals = np.min(fm_vals, axis=1)
    print(f'shape after min {fm_vals.shape}')

    far_vals = compute_fars(fm_vals, far_hist=far_hist)

    ts_farvals = np.linspace(0, 5 / 4096 * len(far_vals), len(far_vals))
    axs[2].set_title('Final metric')
    axs[2].set_xlabel('Time (ms)')
    color = 'black'
    axs[2].set_ylabel('Value, a.u.')
    axs[2].plot(ts_farvals * 1000, fm_vals, label='metric value')
    axs[2].tick_params(axis='y', labelcolor=color)
    axs[2].legend()
    axs[2].set_ylim(-50, 10)

    for i, label in enumerate(SNR_VS_FAR_HL_LABELS):
        if i % 2 == 0:
            metric_val_label = far_to_metric(
                SNR_VS_FAR_HORIZONTAL_LINES[i], far_hist)
            if metric_val_label is not None:
                axs[2].axhline(y=metric_val_label - bias, alpha=0.8**i, label=f'1/{label}', c='black')

    strain = strain[:, 100 + 3 * 5:-(100 + 4 * 5)]

    ts_strain = np.linspace(0, len(strain[0, :]) / 4096, len(strain[0, :]))
    axs[0].set_title(f'{tag} strain, SNR = {snr:.1f}')
    axs[0].plot(ts_strain * 1000, strain[0, :],
                label='Hanford', color=ifo_colors[0])
    axs[0].plot(ts_strain * 1000, strain[1, :],
                label='Livingston', color=ifo_colors[1])
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Whitened strain')
    axs[0].legend()
    axs[0].grid()

    for k in range(len(weights)):
        extracted = np.dot(data, weights[k])

        axs[1].plot(ts_farvals * 1000, extracted,
                    color=colors[k], label=labels[k], linewidth=2)
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Contribution')
    axs[1].grid()
    axs[1].set_title(
        'Per autoencoder final metric contribution + coherence features')
    axs[1].legend()

    xlims = {
        'bbh': (1550, 1550 + 300),
        'sglf': (1550, 1550 + 300),
        'sghf': (1550, 1550 + 300),
        'wnbhf': (2100, 2100 + 300),
        'wnblf': (2100, 2100 + 300),
        'supernova': (2000, 2900)}

    for i in range(3):
        axs[i].set_xlim(xlims[tag])
    a, b = xlims[tag]
    c = b - a
    step = c / 10

    labelLines(axs[2].get_lines(), zorder=2.5, xvals=(
        300, a + step * (1), a + step * (2), a + step * (3),))

    fig.tight_layout()
    axs[0].grid()
    axs[1].grid()
    for i in range(3):
        axs[i].set_xlim(xlims[tag])

    plt.savefig(f'{plot_savedir}/{tag}_3_panel_plot.pdf', dpi=300)


def combined_loss_curves(
    train_losses,
    val_losses,
    tags,
    title,
    savedir,
    show_snr=False):

    centers = CURRICULUM_SNRS
    fig, ax = plt.subplots(1, figsize=(8, 5))
    cols = {
        'BBH': 'steelblue',
        'SG 64-512 Hz': 'salmon',
        'SG 512-1024 Hz': 'goldenrod',
        'Background': 'purple',
        'Glitch': 'darkgreen'
    }
    lines = []
    for k in range(len(train_losses)):
        train_loss, val_loss = train_losses[k], val_losses[k]
        tag = tags[k]
        epoch_count = len(train_loss)

        epochs = np.linspace(1, epoch_count, epoch_count)

        lnt = ax.plot(epochs, np.array(train_loss), linestyle='-', label=f'Training loss, {tag}', c=cols[tag])
        lnv = ax.plot(epochs, np.array(val_loss), linestyle='--', label=f'Validation loss, {tag}', c=cols[tag])
        lines.append(lnt[0])
        lines.append(lnv[0])

    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel('Loss', fontsize=15)
    plt.grid(True)

    if show_snr:
        n_currics = len(CURRICULUM_SNRS)
        ax_1 = ax.twinx()
        for i in range(n_currics):
            low, high = centers[i] - \
                centers[i] // 4, centers[i] + centers[i] // 2
            snr_ln = ax_1.fill_between(epochs[i * epoch_count // n_currics:(
                i + 1) * epoch_count // n_currics + 1], low, high, label='SNR range', color='darkgreen', alpha=0.2)

            if i == 0:
                lines.append(snr_ln)

        ax_1.set_ylabel('SNR range', fontsize=15)
        ax_1.grid()

    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(savedir, dpi=300)


def train_signal_example_plots(
    strain_samples,
    tags,
    savedir,
    snrs=None,
    do_train_sample=True):

    n = len(strain_samples)
    fig, axs = plt.subplots(n, figsize=(8, 5 * n))
    ifos = {0: 'Hanford', 1: 'Livingston'}
    cols = {0: 'goldenrod', 1: 'darkgreen'}
    for i in range(n):
        ts = strain_samples[i].shape[1]
        ts = np.linspace(0, ts * 1 / 4096, ts)
        for j in range(2):
            axs[i].plot(ts * 1000, strain_samples[i]
                        [j, :], c=cols[j], label=ifos[j])

        axs[i].set_xlabel('Time, (ms)')
        axs[i].set_ylabel('Whitened Strain')

        if do_train_sample:
            # show the region for a training sample
            low, high = axs[i].get_ylim()
            start = np.random.uniform(20, 40)
            axs[i].fill_between([start, start + 200 / 4096 * 1000],
                                [low, low], [high, high],
                                color='lightsteelblue', alpha=0.5, label='Example training data')
            axs[i].set_ylim(low, high)
        snr = ''
        if snrs is not None:
            snr = f', SNR: {snrs[i]:.1f}'
        axs[i].set_title(tags[i] + snr)
        if i == 0:
            axs[i].legend()
    fig.tight_layout()
    plt.savefig(savedir, dpi=300)

def learned_fm_weights_colorplot(values, savedir):
    values = np.array(values)
    freq_corr = values[2::3]
    freq_corr = np.sum(freq_corr)

    pearson = values[-1]

    weights = values[:-1].reshape(5, 3)
    weights = weights[:, [True, True, False]]
    minval = min( min(np.min(weights), pearson), freq_corr)
    maxval = max( max(np.max(weights), pearson), freq_corr)
    cmap = plt.colormaps['PiYG_r']
    scale_cmap = cm.get_cmap('PiYG_r', 100)
    def scale(x):
        return scale_cmap((x - minval) / (maxval-minval))

    # weights, freq_corr, pearson
    FONTSIZE = 15
    plt.grid(False)
    fig = plt.figure(figsize=(15, 4), constrained_layout=True)

    widths = [11, 2.25, 0.6]
    heights = [2]
    spec = fig.add_gridspec(ncols=3, nrows=1, width_ratios=widths,
                            height_ratios=heights)
    axs = []
    for row in range(1):
        for col in range(3):
            axs.append(fig.add_subplot(spec[row, col]))

    for elem in axs:
        elem.grid(False)
    fig.canvas.draw()
    axs[0].tick_params(axis='y', labelsize=18, length = 0)
    labels = [item.get_text() for item in axs[0].get_xticklabels()]
    labels[1:] = ["Background", "BBH", "Glitch", " "*0 +"SG 64-512 Hz", " "*15 +"SG 512-1024 Hz"]
    axs[0].set_xticklabels(labels, fontsize=FONTSIZE)

    #$|\widetilde{H_O} \cdot \widetilde{H_R} |$, $|\widetilde{L_O} \cdot \widetilde{L_R} |$, and $|\widetilde{H_R} \cdot \widetilde{L_R} |$
    labels = [item.get_text() for item in axs[0].get_yticklabels()]
    labels = [None, r"$|\widetilde{H_O} \cdot \widetilde{H_R} |$",
                None, r"$|\widetilde{L_O} \cdot \widetilde{L_R} |$"]
    axs[0].set_yticklabels(labels, fontsize=FONTSIZE)
    axs[0].set_title("Autoencoder Features", fontsize=FONTSIZE)
    weight_img = axs[0].imshow(scale(weights.T))


    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)

    cb1 = mpl.colorbar.ColorbarBase(axs[2], cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    axs[2].yaxis.tick_left()
    axs[2].set_yticklabels([item.get_text() for item in axs[2].get_yticklabels()], fontsize=FONTSIZE)
    axs[2].set_title("Scale",fontsize=FONTSIZE)

    corr_img = axs[1].imshow(scale(np.array([freq_corr, pearson]))[:, np.newaxis], alpha=1)
    axs[1].tick_params(axis='both', labelsize=18, length = 0)
    labels = [item.get_text() for item in axs[1].get_yticklabels()]
    print("20", labels)
    labels = [None, r"$|\widetilde{H_O} \cdot \widetilde{L_O} |$", None, "Pearson", None]
    axs[1].set_yticklabels(labels, fontsize=FONTSIZE)
    axs[1].xaxis.set_visible(False)
    axs[1].set_title("Correlation Features", fontsize=FONTSIZE)

    # manually do white line grid
    for x in range(4):
        axs[0].plot([x+0.5, x+0.5], [-0.5, 1.5], c="white", linewidth=6)
    for y in range(1):
        axs[0].plot([-0.5, 4.5], [0.5+y, 0.5+y], c="white", linewidth=6)

    axs[1].plot([-0.5, 0.5], [0.5, 0.5], c="white", linewidth=6)

    for y in range(2):
        for x in range(5):
            axs[0].text(x-0.15, y+0.05, f"{weights.T[y, x]:.2f}", fontsize=14)

    for y in range(2):
        axs[1].text( -0.15, y+0.05, f"{np.array([freq_corr, pearson])[y]:.2f}", fontsize=14)

    try:
        fig.tight_layout()
        None
    except RuntimeError:
        None

    plt.savefig(savedir, bbox_inches='tight',dpi=300)

def make_roc_curves(
        datas,
        amp_measures,
        metric_coefs,
        far_hist,
        tags,
        savedir,
        special,
        bias,
        smoothing_window=5,
        hrss=False,
        MLy_colors=False,
        done_fm_evals = None):
    fig, axs = plt.subplots(1, figsize=(12, 8))
    colors = {
        'bbh': 'steelblue',
        'sglf': 'salmon',
        'sghf': 'goldenrod',
        'wnbhf': 'purple',
        'wnblf': 'hotpink',
        'supernova': 'darkorange'
    }
    if not hrss:
        axs.set_xlabel(f'SNR', fontsize=20)
    else:
        axs.set_xlabel(f'hrss', fontsize=20)

    axs.set_ylabel('Fraction of events detected at FAR 1/year', fontsize=20)
    #print(528, bias)
    data_for_plotting = dict()
    for k in range(len(datas)):
        data = datas[k]
        amp_measure = amp_measures[k]
        tag = tags[k]
        print('plotting', tag)

        # if done_fm_evals == None:
        #     if RETURN_INDIV_LOSSES:
        #         fm_vals = metric_coefs(torch.from_numpy(
        #             data).float().to(DEVICE)).detach().cpu().numpy()
        #     else:
        #         fm_vals = np.dot(data, metric_coefs) + bias
        #     print(539, fm_vals.shape)
        #     if smoothing_window != 1:
        #         fm_vals = np.apply_along_axis(
        #             lambda m: np.convolve(m, np.ones(smoothing_window)/smoothing_window, mode='same'),
        #             axis=1,
        #             arr=fm_vals)

        #     fm_vals = np.min(fm_vals, axis=1)
        #     assert False
        # else:
        #     print(554, "loading from previous")
        fm_vals = done_fm_evals[tag]
        print(tag, fm_vals[:100])

        print(546, fm_vals.shape)
        rename_map = {
            'background': 'Background',
            'bbh': 'BBH',
            'glitches': 'Glitch',
            'sglf': 'SG 64-512 Hz',
            'sghf': 'SG 512-1024 Hz',
            'wnblf': 'WNB 40-400 Hz',
            'wnbhf': 'WNB 400-1000 Hz',
            'supernova': 'Supernova'
        }
        tag_ = rename_map[tag]
        metric_val_label = far_to_metric(
            365*24*3600, far_hist) #

        # this is calibrated 1/year FM
        metric_val_label = -2.4299999999999997

        am_bins = np.arange(0, VARYING_SNR_HIGH, 1)

        nbins = len(am_bins)
        am_bin_detected = [0]*nbins
        am_bin_total = [0]*nbins
        for i, am in enumerate(amp_measure):
            insert_location = np.searchsorted(am_bins, am)

            if insert_location >= 200:
                continue
            am_bin_total[insert_location] += 1
            detec_stat = fm_vals[i]
            if detec_stat <= metric_val_label:
                am_bin_detected[insert_location] += 1

        TPRs = []
        TPRs_errors = []
        snr_bins_plot = []
        for i in range(nbins):
            if am_bin_total[i] != 0:
                TPR = am_bin_detected[i]/am_bin_total[i]
                TPRs.append(TPR)
                TPRs_errors.append( np.sqrt(TPR * (1 - TPR) / am_bin_total[i]) )
                snr_bins_plot.append(am_bins[i]) #only adding it if nonzero total in that bin

        axs.plot(snr_bins_plot, TPRs, label=tag, c=colors[tag])
        data_for_plotting[tag] = [snr_bins_plot, TPRs, TPRs_errors]

    np.savez('scripts/plots/eff_after_heuristics.npz', **data_for_plotting)
    axs.legend()
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(f'{savedir}/{special}.pdf', dpi=300)
    # plt.show()
    plt.close()

def make_roc_curves_smoothing_comparison(data,
        amp_measure,
        metric_coefs,
        far_hist,
        tag,
        savedir,
        special,
        bias,
        smoothing_windows,
        hrss=False,
        MLy_colors=False):

    fig, axs = plt.subplots(1, figsize=(12, 8))
    colors = {
        'bbh': 'blue',
        'sg': 'red',
        'sglf': 'red',
        'sghf': 'orange',
        'wnbhf': 'darkviolet',
        'wnblf': 'deeppink',
        'supernova': 'goldenrod'
    }
    if MLy_colors:
        colors = {
            'bbh': "brown",
            'wnbhf' : "blue",
            "wnblf" : "lightblue",
            "sghf" : "darkgreen",
            "sglf" : "green",
            "supernova" : "orange"
        }
    if not hrss:
        axs.set_xlabel(f'SNR', fontsize=20)
    else:
        axs.set_xlabel(f'hrss', fontsize=20)

    axs.set_ylabel('Fraction of events detected at FAR 1/year', fontsize=20)

    if RETURN_INDIV_LOSSES:
        fm_vals = metric_coefs(torch.from_numpy(
            data).float().to(DEVICE)).detach().cpu().numpy()
    else:
        fm_vals = np.dot(data, metric_coefs)

    fm_vals_orig = fm_vals[:]
    for smoothing_window in smoothing_windows:
        if smoothing_window != 1:
            fm_vals = np.convolve(fm_vals_orig, np.ones(smoothing_window)/smoothing_window, mode='valid')

        fm_vals = np.min(fm_vals, axis=1)

        rename_map = {
            'background': 'Background',
            'bbh': 'BBH',
            'glitches': 'Glitch',
            'sglf': 'SG 64-512 Hz',
            'sghf': 'SG 512-1024 Hz',
            'wnblf': 'WNB 40-400 Hz',
            'wnbhf': 'WNB 400-1000 Hz',
            'supernova': 'Supernova'
        }
        tag_ = rename_map[tag]
        metric_val_label = far_to_metric(
            365*24*3600, far_hist) #

        # positive detection are the ones below the curve
        if not hrss:
            am_bins = np.arange(0, VARYING_SNR_HIGH, 1)
        else:
            axs.set_xscale("log")
            am_bins = np.logspace(-23, -18, 200) # not yet clear what this will look like

        nbins = len(am_bins)
        am_bin_detected = [0]*nbins
        am_bin_total = [0]*nbins
        for i, am in enumerate(amp_measure):
            insert_location = np.searchsorted(am_bins, am)

            if insert_location >= 200:
                continue

            am_bin_total[insert_location] += 1
            detec_stat = fm_vals[i]
            if detec_stat <= metric_val_label:
                am_bin_detected[insert_location] += 1

        TPRs = []
        snr_bins_plot = []
        for i in range(nbins):
            if am_bin_total[i] != 0:
                TPRs.append(am_bin_detected[i]/am_bin_total[i])
                snr_bins_plot.append(am_bins[i]) #only adding it if nonzero total in that bin

        if tag in colors:
            axs.plot(snr_bins_plot, TPRs, label = f"{tag}, window: {smoothing_window}", c=colors[tag], alpha=(1-smoothing_window/200)**2)
        else:
            axs.plot(snr_bins_plot, TPRs, label = f"{tag}, window: {smoothing_window}", alpha=(1-smoothing_window/200)**2)

    axs.legend()
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(f'{savedir}/{special}.pdf', dpi=300)
    plt.show()
    plt.close()

def make_roc_curves_smoothing_comparison(data,
        amp_measure,
        metric_coefs,
        far_hist,
        tag,
        savedir,
        special,
        bias,
        smoothing_windows,
        hrss=False,
        MLy_colors=False):

    fig, axs = plt.subplots(1, figsize=(12, 8))
    colors = ['steelblue', 'salmon', 'goldenrod','purple',  'hotpink', 'darkorange']

    if not hrss:
        axs.set_xlabel(f'SNR', fontsize=20)
    else:
        axs.set_xlabel(f'hrss', fontsize=20)

    axs.set_ylabel('Fraction of events detected at FAR 1/year', fontsize=20)

    if RETURN_INDIV_LOSSES:
        fm_vals = metric_coefs(torch.from_numpy(
            data).float().to(DEVICE)).detach().cpu().numpy()
    else:
        fm_vals = np.dot(data, metric_coefs)

    fm_vals_orig = fm_vals[:]
    for i_window, smoothing_window in enumerate(smoothing_windows):

        far_hist = np.load(f'{args.data_predicted_path}/far_bins_{smoothing_window}.npy')
        if smoothing_window != 1:
            fm_vals = np.apply_along_axis(lambda m: np.convolve(m, np.ones(smoothing_window)/smoothing_window, mode='same'),
                axis=1,
                arr=fm_vals_orig)

        fm_vals = np.min(fm_vals, axis=1)

        rename_map = {
            'background': 'Background',
            'bbh': 'BBH',
            'glitches': 'Glitch',
            'sglf': 'SG 64-512 Hz',
            'sghf': 'SG 512-1024 Hz',
            'wnblf': 'WNB 40-400 Hz',
            'wnbhf': 'WNB 400-1000 Hz',
            'supernova': 'Supernova'
        }
        tag_ = rename_map[tag]
        metric_val_label = far_to_metric(
            365*24*3600, far_hist) #

        #positive detection are the ones below the curve
        if not hrss:
            am_bins = np.arange(0, VARYING_SNR_HIGH, 1)
        else:
            axs.set_xscale("log")
            am_bins = np.logspace(-23, -18, 200) # not yet clear what this will look like

        nbins = len(am_bins)
        am_bin_detected = [0]*nbins
        am_bin_total = [0]*nbins
        for i, am in enumerate(amp_measure):
            insert_location = np.searchsorted(am_bins, am)

            if insert_location >= 200:
                continue

            am_bin_total[insert_location] += 1
            detec_stat = fm_vals[i]
            if detec_stat <= metric_val_label:
                am_bin_detected[insert_location] += 1

        TPRs = []
        snr_bins_plot = []
        for i in range(nbins):
            if am_bin_total[i] != 0:
                TPRs.append(am_bin_detected[i]/am_bin_total[i])
                snr_bins_plot.append(am_bins[i]) #only adding it if nonzero total in that bin
        if tag in colors:
            axs.plot(snr_bins_plot, TPRs,
                label=f'window: {smoothing_window}',
                color=colors[i_window])
            axs.set_title(tag_, fontsize=20)
        else:
            axs.plot(snr_bins_plot, TPRs,
                label=f'window: {smoothing_window}',
                color=colors[i_window])
            axs.set_title(tag_, fontsize=20)

    axs.legend(fontsize=20)
    plt.grid(True)
    fig.tight_layout()
    plt.savefig(f'{savedir}/{special}.pdf', dpi=300)


def main(args):

    model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
    model.load_state_dict(torch.load(
        args.fm_model_path, map_location=GPU_NAME))
    weight = np.concatenate([model.layer.weight.data.cpu().numpy()[0], np.array([0])])
    learned_dp_weights = weight[:]

    linear_weight = model.layer.weight.data.cpu().numpy()
    print(weight)
    """
        Factors to keep for the FM
        0 - background AE (L_O * L_R)
        1 - background AE (H_O * H_R)
        2 - background AE (L_O * H_O)
        3 - BBH AE (L_O * L_R)
        4 - BBH AE (H_O * H_R)
        5 - BBH AE (L_O * H_O)
        6 - Glitches AE (L_O * L_R)
        7 - Glitches AE (H_O * H_R)
        8 - Glitches AE (L_O * H_O)
        9 - SGLF AE (L_O * L_R)
        10 - SGLF AE (H_O * H_R)
        11 - SGLF AE (L_O * H_O)
        12 - SGHF AE (L_O * L_R)
        13 - SGHF AE (H_O * H_R)
        14 - SGHF AE (L_O * H_O)
        15 - Pearson
    """

    bias = model.layer.bias.data.cpu().numpy()[0]
    print('bias!:', bias)

    print(f'Learned weights: {learned_dp_weights}')
    weights = []

    for i in range(5):
        arr = np.zeros(weight.shape)
        arr[3*i] = weight[3*i]
        arr[3*i+1] = weight[3*i+1]
        weights.append(arr)

    # shared, original -> original coefficient
    arr = np.zeros(weight.shape)
    for i in range(5):
        arr[3*i+2] = weight[3*i+2]
    weights.append(arr)

    # pearson coefficient
    arr = np.zeros(weight.shape)
    arr[-1] = weight[-1]
    weights.append(arr)

    type1 = False
    do_snr_vs_far = 0
    do_fake_roc = type1
    do_3_panel_plot = type1
    do_combined_loss_curves = type1
    do_train_signal_example_plots = type1
    do_anomaly_signal_show = type1
    do_learned_fm_weights = 0
    do_make_roc_curves = 1
    do_heuristic_efficiency = 1

    if do_snr_vs_far or do_make_roc_curves:

        metric_coefs = np.load(f'{args.data_predicted_path}/trained/final_metric_params.npy')
        means, stds = np.load(f'{args.data_predicted_path}/trained/norm_factor_params.npy')
        tags = ['bbh', 'wnbhf', 'supernova', 'wnblf', 'sglf', 'sghf']


        if RETURN_INDIV_LOSSES:
            model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
            model.load_state_dict(torch.load(
                args.fm_model_path, map_location=GPU_NAME))

        data_dict = {}
        snrs_dict = {}
        hrss_dict = {}
        # for tag in tags:

        #     print(f'loading {tag}')
        #     ts = time.time()
        #     data = np.load(f'{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy')

        #     print(f'{tag} loaded in {time.time()-ts:.3f} seconds')

        #     data = (data - means) / stds
        #     data = data#[1000:]

        #     snrs = np.load(f"{args.data_predicted_path}/data/{tag}_varying_snr_values.npy")
        #     data_dict[tag] = data
        #     snrs_dict[tag] = snrs

        X3 = ['bbh', 'sglf', 'sghf', 'wnbhf', 'supernova', 'wnblf']

        # far_hist = np.load(f'{args.data_predicted_path}/far_bins_{SMOOTHING_KERNEL}.npy')

        # done_fm_evals = amp_measure_vs_far_plotting([data_dict[elem] for elem in X3],
        #                     [snrs_dict[elem] for elem in X3],
        #                     model,
        #                     far_hist,
        #                     X3,
        #                     args.plot_savedir,
        #                     f'Detection Efficiency, SNR, window: {SMOOTHING_KERNEL}',
        #                     bias)

        if do_heuristic_efficiency:
            fm_model_path = (f"{args.data_predicted_path}/trained/fm_model.pt")
            fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
            fm_model.load_state_dict(torch.load(
                fm_model_path, map_location=GPU_NAME))

            linear_weights = fm_model.layer.weight.detach()#.cpu().numpy()
            bias_value = fm_model.layer.bias.detach()#.cpu().numpy()

            mean_norm = torch.from_numpy(means).to(DEVICE)#[:-1]
            std_norm = torch.from_numpy(stds).to(DEVICE)#[:-1]

            tags = ['bbh', 'sghf', 'wnbhf', 'supernova', 'wnblf', 'sglf', 'sghf']
            for tag in tags:

                # print(f'loading {tag}')
                # ts = time.time()
                # data = np.load(f'{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy')
                # data = torch.from_numpy(data).to(DEVICE).float()

                # print(f'{tag} loaded in {time.time()-ts:.3f} seconds')

                # data = (data - means) / stds
                # data = data#[1000:]
                # snrs = np.load(f'{args.data_predicted_path}/data/{tag}_varying_snr_SNR.npz.npy')#[1000:]
                snrs = np.load(f'output/O3av2/evaluated/heuristic/SIG_EVAL{tag}_SNRs.npy')
                # passed_herustics = np.load(f'{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals_heuristic_res.npy')
                passed_herustics = np.load(f'output/O3av2/evaluated/heuristic/SIG_EVAL_{tag}_heur_model_evals.npy')

                data_dict[tag] = passed_herustics
                snrs_dict[tag] = snrs


        if do_make_roc_curves: #roc curve

            far_hist = np.load(f'{args.data_predicted_path}/far_bins_{SMOOTHING_KERNEL}.npy')

            if RETURN_INDIV_LOSSES:
                model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
                model.load_state_dict(torch.load(
                    args.fm_model_path, map_location=GPU_NAME))

            make_roc_curves([data_dict[elem] for elem in X3],
                            [snrs_dict[elem] for elem in X3],
                            model,
                            far_hist,
                            X3,
                            args.plot_savedir,
                            f'ROC plots, SNR, window: {SMOOTHING_KERNEL}',
                            bias,
                            smoothing_window=SMOOTHING_KERNEL,
                            done_fm_evals=data_dict)
            # make_roc_curves([data_dict[elem] for elem in X3],
            #                     [hrss_dict[elem] for elem in X3],
            #                     model,
            #                     far_hist,
            #                     X3,
            #                     args.plot_savedir,
            #                     f'ROC plots, hrss, window: {SMOOTHING_KERNEL}',
            #                     bias,
            #                     smoothing_window=SMOOTHING_KERNEL,
            #                     hrss=True)

            # # make the plot showing all the smoothing windows for a single class at once
            # for elem in X3:
            #     make_roc_curves_smoothing_comparison(data_dict[elem],
            #                         snrs_dict[elem],
            #                         model,
            #                         far_hist,
            #                         elem,
            #                         args.plot_savedir,
            #                         f'ROC-plots_SNR_vary_smoothing_{elem}',
            #                         bias,
            #                         SMOOTHING_KERNEL_SIZES)


            # # make the plot showing all the smoothing windows for a single class at once
            # for elem in X3:
            #     make_roc_curves_smoothing_comparison(data_dict[elem],
            #                         snrs_dict[elem],
            #                         model,
            #                         far_hist,
            #                         elem,
            #                         args.plot_savedir,
            #                         f'ROC plots, SNR, vary smoothing, {elem}',
            #                         bias,
            #                         SMOOTHING_KERNEL_SIZES)

    if do_fake_roc:

        far_hist = np.load(f'{args.data_predicted_path}/far_bins_{SMOOTHING_KERNEL}.npy')
        fake_roc_plotting(far_hist, args.plot_savedir)

    if do_3_panel_plot:

        far_hist = np.load(f'{args.data_predicted_path}/far_bins_{SMOOTHING_KERNEL}.npy')
        metric_coefs = np.load(f'{args.data_predicted_path}/trained/final_metric_params.npy')
        norm_factors = np.load(f'{args.data_predicted_path}/trained/norm_factor_params.npy')
        means, stds = norm_factors[0], norm_factors[1]

        tags = ['bbh', 'sghf', 'sglf', 'wnbhf', 'supernova', 'wnblf']
        inds = {
            'bbh': 0,
            'sghf': 0,
            'sglf': 0,
            'wnbhf': 1256,
            'wnblf': 1958,
            'supernova': 1228
        }
        for tag in tags:
            strains = np.load(f'{args.data_predicted_path}/data/{tag}_varying_snr.npz')['data'][:, inds[tag]]
            data = np.load(f'{args.data_predicted_path}/evaluated/{tag}_varying_snr_evals.npy')[inds[tag]]
            # data = np.delete(data, FACTORS_NOT_USED_FOR_FM, -1)
            data = (data - means) / stds
            snrs = np.load(f'{args.data_predicted_path}/data/{tag}_varying_snr_SNR.npz.npy')[inds[tag]]

            if RETURN_INDIV_LOSSES:
                model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
                model.load_state_dict(torch.load(
                    args.fm_model_path, map_location=GPU_NAME))
                three_panel_plotting(
                    strains, data, snrs, model, far_hist, tag, args.plot_savedir, bias, weights)
            else:
                three_panel_plotting(
                    strains, data, snrs, metric_coefs, far_hist, tag, args.plot_savedir, bias, weights)

    if do_combined_loss_curves:
        load_path = f'{args.data_predicted_path}/trained/'

        signal_classes = ['bbh', 'sglf', 'sghf']
        tags = ['BBH', 'SG 64-512 Hz', 'SG 512-1024 Hz']
        train_losses = []
        val_losses = []
        for sc in signal_classes:
            train_losses.append(np.load(f'{load_path}/{sc}/loss_hist.npy'))
            val_losses.append(np.load(f'{load_path}/{sc}/val_loss_hist.npy'))

        combined_loss_curves(train_losses, val_losses, tags,
                             'Signals loss curves',
                             f'{args.plot_savedir}/signal_loss_curve.pdf',
                             show_snr=True)

        bkg_classes = ['background', 'glitches']
        tags = ['Background', 'Glitch']
        train_losses = []
        val_losses = []
        for bc in bkg_classes:
            train_losses.append(np.load(f'{load_path}/{bc}/loss_hist.npy'))
            val_losses.append(np.load(f'{load_path}/{bc}/val_loss_hist.npy'))

        combined_loss_curves(train_losses, val_losses, tags,
                             'Non-signals loss curves',
                             f'{args.plot_savedir}/backgrounds_loss_curve.pdf')

    if do_train_signal_example_plots:
        inds = {
            'bbh': 0,
            'sglf': 3,
            'sghf': 113
        }
        tags = list(inds.keys())
        strains = []
        snrs = []
        ind = 1
        for tag in tags:
            strain = np.load(f'{args.data_predicted_path}/data/{tag}_varying_snr.npz', mmap_mode='r')['data'][inds[tag]][:, int((1680 + 50) * 4.096):int(4.096 * (1880 - 50))]
            snr = np.load(f'{args.data_predicted_path}/data/{tag}_varying_snr_SNR.npz.npy', mmap_mode='r')[inds[tag]]
            strains.append(strain)
            snrs.append(snr)

        train_signal_example_plots(strains,
                                   ['BBH', 'SG 64-512Hz', 'SG 512-1024Hz'],
                                   f'{args.plot_savedir}/signal_train_exs.pdf',
                                   snrs)

        strains = []
        timeslides = np.load(f'{args.data_predicted_path}/data/timeslides.npz', mmap_mode='r')['data']

        a = int(340740 * 4.096)
        b = int(a + 100 * 4.096)
        glitch = timeslides[:, a:b]
        strains.append(glitch)

        a = int(287489 * 4.096)  # (i put a random number)
        b = int(a + 100 * 4.096)
        bkg = timeslides[:, a:b]
        strains.append(bkg)

        train_signal_example_plots(strains,
                                   ['Glitch', 'Background'],
                                   f'{args.plot_savedir}/background_train_exs.pdf')

    if do_anomaly_signal_show:
        tags = ['wnbhf', 'wnblf', 'supernova']
        strain_data = []
        for tag in tags:
            data = np.load(f'{args.data_predicted_path}/data/{tag}.npz')['data']
            sample = data[-1, :, 0, int(
                (1000 - 50) * 4.096):int((1000 + 50) * 4.096)]
            strain_data.append(sample)

        train_signal_example_plots(strain_data,
                                   ['WNB 400-1000Hz', 'WNB 40-400Hz', 'Supernova'],
                                   f'{args.plot_savedir}/anomaly_exs.pdf',
                                   do_train_sample=False)

    if do_learned_fm_weights:
        learned_fm_weights_colorplot(learned_dp_weights,
            f'{args.plot_savedir}/learned_fm_weights.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_predicted_path', help='Path to model directory',
                        type=str)

    parser.add_argument('plot_savedir', help='Required output directory for saving plots',
                        type=str)

    parser.add_argument('fm_model_path', help='Path to the final model',
                        type=str)

    args = parser.parse_args()
    main(args)
