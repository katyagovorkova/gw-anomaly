import argparse
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import process_linear_fm
import torch
from config import (GPU_NAME,VARYING_SNR_LOW, VARYING_SNR_HIGH)
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
from scipy.signal import welch

def get_mean_std_line(snr_bins, gen_snrs, y):
    bins = {}
    for snr in snr_bins:
        bins[snr] = []

    for i, elem in enumerate(gen_snrs):
        bins[elem].append(y[i])

    means, stds = np.zeros(snr_bins.shape), np.zeros(snr_bins.shape)
    for i, snr in enumerate(snr_bins):
        means[i] = np.mean(bins[snr])
        stds[i] = np.std(bins[snr])

    return means, stds

def compute_lost_fraction(snr_bins, generated_snr, freq_corr_contrib, corr_cut):
    bins = {}
    for snr in snr_bins:
        bins[snr] = []
        
    for i, elem in enumerate(generated_snr):
        bins[elem].append(freq_corr_contrib[i])

    res = np.zeros(snr_bins.shape)
    for i, snr in enumerate(snr_bins):
        res[i] = (np.array(bins[snr])>corr_cut).sum() / len(bins[snr])

    return res

def pearson(Hpsd, Lpsd):
    H_ = Hpsd - Hpsd.mean()
    L_ = Lpsd - Lpsd.mean()
    return np.sum(H_*L_) / (np.sum(H_**2)*np.sum(L_**2))**0.5

def main(args):
    try:
        os.makedirs(args.save_path)
    except FileExistsError:
        None
    DEVICE = torch.device(GPU_NAME)

    strain_data = np.load(args.generated_data_path)['data']
    generated_snr = np.load(args.generated_data_path[:-4] + "_SNR.npz.npy")  
    data_eval = np.load(args.eval_data_path)

    fm_model_path = f"{MODELS_LOCATION}/fm_model.pt"
    norm_factors_path = f"{MODELS_LOCATION}/norm_factor_params.npy"

    scaled_evals, unscaled_evals, fm_vals, (weight, bias) = process_linear_fm(data_eval, fm_model_path, norm_factors_path, DEVICE)
    
    best_fm_vals = np.min(fm_vals, axis=1)
    best_locs = np.argmin(fm_vals, axis=1)[:, 0]
    
    #compute an average "background" PSD
    
    f, H_bkg = welch(strain_data[:, 0, 2049:2049+200], fs=4096)
    L_bkg = welch(strain_data[:, 1, 2049:2049+200], fs=4096)[1]
    H_bkg = np.mean(H_bkg, axis=0)
    L_bkg = np.mean(L_bkg, axis=0)

    plt.plot(f, H_bkg)
    plt.plot(f, L_bkg)
    plt.savefig(f"{args.save_path}/base_psds.png", dpi=300)
    plt.close()


    #print("H_bkg", H_bkg)

    L = strain_data.shape[2] // 2 #+ 2000
    smin, smax = L+1000-200, L + 1000
    print("smin, smax", smin, smax)
    strain_data = strain_data[:, :, smin:smax]
    print(strain_data.shape)
    #strain_data = np.random.normal(0, 1, strain_data.shape)
    print("64", strain_data[:, 0, :].shape, strain_data[:, 1, :].shape)
    Hpsds = welch(strain_data[:, 0, :], fs=4096)[1] - H_bkg
    Lpsds = welch(strain_data[:, 1, :], fs=4096)[1] - L_bkg
    print("67", Hpsds.shape, Lpsds.shape)

    #plt.plot(strain_data[0, 0])
    #plt.plot(strain_data[0, 1])
    #plt.savefig(f"{args.save_path}/strain0_{generated_snr[0]}.png", dpi=300)
    #plt.show()
    for i in range(10):
        plt.plot(strain_data[i, 0])
        plt.plot(strain_data[i, 1])
        plt.savefig(f"{args.save_path}/strain0_{generated_snr[i]}.png", dpi=300)
        plt.close()

        plt.plot(Hpsds[i])
        plt.plot(Lpsds[i])
        plt.savefig(f"{args.save_path}/psd0_{generated_snr[i]}.png", dpi=300)
        plt.close()


    psd_scores = []
    for i in range(len(Hpsds)):
        #psd_scores.append(pearson(Hpsds[i], Lpsds[i]))
        psd_scores.append(np.log10(Hpsds[i].sum())+np.log10(Lpsds[i].sum()))
    psd_scores = np.array(psd_scores)

    best_scaled_evals = []
    best_unscaled_evals = []
    for i in range(len(best_locs)):
        best_scaled_evals.append(scaled_evals[i, best_locs[i], :])
        best_unscaled_evals.append(unscaled_evals[i, best_locs[i], :])
    best_scaled_evals = np.stack(best_scaled_evals, axis=1).T
    best_unscaled_evals = np.stack(best_unscaled_evals, axis=1).T

    freq_corr_contrib = best_scaled_evals[:, 2]
    for x in [5, 8, 11, 14]:
        freq_corr_contrib += best_scaled_evals[:, x]
    
    snr_bins = np.arange(VARYING_SNR_LOW, VARYING_SNR_HIGH, 1)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    axs[0, 0].scatter(generated_snr, psd_scores)
    axs[0, 0].set_ylabel("PSD feature")

    axs[0, 1].scatter(generated_snr, freq_corr_contrib)
    axs[0, 1].set_ylabel("scaled freq corr")
    means, stds = get_mean_std_line(snr_bins, generated_snr, freq_corr_contrib)
    axs[0, 1].plot(snr_bins, means, c="black", linewidth=3)
    axs[0, 1].fill_between(snr_bins, means-stds*0.5, means+stds*0.5, alpha=0.4, color="red")

    axs[0, 2].scatter(generated_snr, best_unscaled_evals[:, -1])
    axs[0, 2].set_ylabel("max pearson corr")
    means, stds = get_mean_std_line(snr_bins, generated_snr, best_unscaled_evals[:, -1])
    axs[0, 2].plot(snr_bins, means, c="black", linewidth=3)
    axs[0, 2].fill_between(snr_bins, means-stds*0.5, means+stds*0.5, alpha=0.4, color="red")

    axs[1, 0].scatter(generated_snr, best_scaled_evals[:, 3])
    axs[1, 0].set_xlabel("SNR")
    axs[1, 0].set_ylabel("Livingston BBH feature")
    means, stds = get_mean_std_line(snr_bins, generated_snr, best_scaled_evals[:, 3])
    axs[1, 0].plot(snr_bins, means, c="black", linewidth=3)
    axs[1, 0].fill_between(snr_bins, means-stds*0.5, means+stds*0.5, alpha=0.4, color="red")


    axs[1, 1].scatter(generated_snr, best_scaled_evals[:, 4])
    axs[1, 1].set_xlabel("SNR")
    axs[1, 1].set_ylabel("Hanford BBH feature")
    means, stds = get_mean_std_line(snr_bins, generated_snr, best_scaled_evals[:, 4])
    axs[1, 1].plot(snr_bins, means, c="black", linewidth=3)
    axs[1, 1].fill_between(snr_bins, means-stds*0.5, means+stds*0.5, alpha=0.4, color="red")

    # plot of BBH samples lost by making the cut
    # generated_snr, freq_corr_contrib
    corr_cut = -0.2
    frac_lost = compute_lost_fraction(snr_bins, generated_snr, freq_corr_contrib, corr_cut)
    axs[1, 2].plot(snr_bins, frac_lost)
    axs[1, 2].set_xlabel("SNR")
    axs[1, 2].set_ylabel("Fraction lost")
    axs[1, 2].set_title(f"cut of < {corr_cut}")

    name = args.generated_data_path[:-4].split("/")[-1].split("_")[0]
    fig.suptitle(f"Cutting efficiency for {name}")

    
    fig.tight_layout()
    plt.savefig(f"{args.save_path}/3x2_pannel.png", dpi=300)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('eval_data_path', type=str,
                        help='Directory containing the evaluated_injections')

    parser.add_argument('save_path', type=str,
                        help='Folder to which save the plots')

    parser.add_argument('generated_data_path', type=str,
                        help='Location of the generated data, for getting SNR')

    args = parser.parse_args()
    main(args)
