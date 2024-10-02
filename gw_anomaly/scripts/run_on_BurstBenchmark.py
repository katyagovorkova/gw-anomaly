import os
import numpy as np
import argparse
from gwpy.timeseries import TimeSeries
import torch
from astropy import units as u
import matplotlib.pyplot as plt
from models import LinearModel, GwakClassifier
from evaluate_data import full_evaluation
import json
import matplotlib
import tqdm
from config import (
    CHANNEL,
    GPU_NAME,
    SEGMENT_OVERLAP,
    SAMPLE_RATE,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    FACTORS_NOT_USED_FOR_FM,
    MODELS_LOCATION
    )

from helper_functions import far_to_metric, compute_fars
DEVICE = torch.device(GPU_NAME)


def clustering(x, bar=15*4096/5):

    # 15 second spacing between events

    cluster = []
    cluster.append(x[0])

    for i, point in enumerate(x[1:]):
        if x[i] - x[i-1] > bar:
            cluster.append(point)

    return cluster


def whiten_bandpass_resample(
        file,
        savedir,
        sample_rate=SAMPLE_RATE,
        bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH,
        shift=None):

    device = torch.device(GPU_NAME)

    fnL1 = "/scratch/florent.robinet/BurstBenchmark/L-L1_%s"%file
    fnH1 = "/scratch/florent.robinet/BurstBenchmark/H-H1_%s"%file
    print(fnL1, fnH1)
    start_point = file.split("-")[1]
    end_point = int(start_point) + int(file.split("-")[2].split(".")[0])

    # Load LIGO data
    strainL1 = TimeSeries.read(fnL1, "L1:STRAIN_BURST_4")
    strainH1 = TimeSeries.read(fnH1, "H1:STRAIN_BURST_4")
    t0 = int(strainL1.t0 / u.s)
    print(t0)

    if shift != None:
        shift_datapoints = shift * sample_rate
        temp = strainL1[-shift_datapoints:]
        strainL1[shift_datapoints:] = strainL1[:-shift_datapoints]
        strainL1[:shift_datapoints] = temp

    # Whiten, bandpass, and resample
    strainL1 = strainL1.resample(sample_rate).bandpass(bandpass_low, bandpass_high)
    strainL1 = strainL1.whiten()

    strainH1 = strainH1.resample(sample_rate).bandpass(bandpass_low, bandpass_high)
    strainH1 = strainH1.whiten()

    return [strainH1, strainL1], start_point

def get_evals(data, trained_path, savedir, start_point, extra = "", far_search=None):
    strain = np.stack(data, axis=1)[np.newaxis, :, :]
    strain_orig = np.stack(data, axis=1)
    strain = np.swapaxes(strain, 1, 2) # make it (N_batches, 2, time_axis), N_batches is 1 here

    #break the strain into pieces along the time axis to fit into GPU memory
    max_gpu_seconds = 50 # 500 seconds at a time
    max_gpu_dtps = max_gpu_seconds * SAMPLE_RATE

    n_splits = (strain.shape[2] // max_gpu_dtps) + 1
    split_data = []
    for n in range(n_splits):
        split_data.append(strain[:, :, n*max_gpu_dtps:(n+1)*max_gpu_dtps])


    model_path = f"{trained_path}/models/"
    model_paths = []
    for elem in ["background.pt", "bbh.pt",  "glitches.pt",
                 "sghf.pt", "sglf.pt"]:
        model_paths.append(f'{model_path}/{elem}')


    norm_factors = np.load(f"{trained_path}/norm_factor_params.npy")

    device=DEVICE
    evals = []
    midpoints = []
    for i, strain_elem in enumerate(split_data):
        ev, mdp = full_evaluation(strain_elem, model_paths, device, return_midpoints=True)
        ev = ev.cpu().detach().numpy()
        ev = np.delete(ev, FACTORS_NOT_USED_FOR_FM, -1)
        evals.append(ev)
        midpoints.append(mdp+i*(max_gpu_dtps))


    fm_model_path = (f"{trained_path}/fm_model.pt")

    fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
    fm_model.load_state_dict(torch.load(
        fm_model_path, map_location=GPU_NAME))#map_location=f'cuda:{args.gpu}'))
    linear_weights = fm_model.layer.weight.detach().cpu().numpy()
    bias_value = fm_model.layer.bias.detach().cpu().numpy()

    scores = []
    scaled_evals = []
    for elem in evals:
        elem = (elem - norm_factors[0]) / norm_factors[1]
        scaled_eval = np.multiply(elem, linear_weights)
        #assert 0
        scaled_evals.append(scaled_eval[0, :, :])
        elem = torch.from_numpy(elem).to(DEVICE)
        scores.append(fm_model(elem).detach().cpu().numpy()[0, :, 0])

    # do smoothing on the scores
    kernel_len = 50
    kernel = np.ones(kernel_len)/kernel_len
    #kernel_evals = np.ones((kernel_len, scaled_evals[0].shape[1]))/kernel_len
    scores_ = []
    bottom_trim = kernel_len * 5
    top_trim = - bottom_trim
    for elem in scores:
        scores_.append(np.convolve(elem, kernel, mode='same')[bottom_trim:top_trim])
    scaled_evals_ = []
    for elem in scaled_evals:
        scaled_evals_.append(np.apply_along_axis(lambda m : np.convolve(m, kernel, mode='same')[bottom_trim:top_trim], axis=0, arr=elem))
    scaled_evals = scaled_evals_
    scaled_evals = np.concatenate(scaled_evals, axis=0) #same indexing as scores
    #for elem in
    scores = scores_
    midpoints_ = []
    for elem in midpoints:
        midpoints_.append(elem[bottom_trim:top_trim])
    midpoints = midpoints_
    #print(scaled_evals[0].shape)

    scaled_evals_ = np.zeros((scaled_evals.shape[0], scaled_evals.shape[1]-4))
    counter = 0
    frequency_correlation = np.zeros(scaled_evals.shape[0])
    for i in range(scaled_evals.shape[1]):
        if i % 3 == 2:
            #frequeny correlation value
            frequency_correlation += scaled_evals[:, i]
        else:
            scaled_evals_[:, counter] = scaled_evals[:, i]
            counter += 1

    scaled_evals_[:, -1] = frequency_correlation



    far_bins = np.load(f"{trained_path}/far_bins_k{kernel_len}.npy")

    # set a bar at 1/2 days FAR to get elements of interest
    far_bar = far_to_metric(3600*24*2, far_bins)

    #print("1/2 day bar", far_bar)
    all_midpoints = np.hstack(midpoints)
    #passed_scores = []
    passed_midpoints = []
    for i in range(len(scores)):
        sc = scores[i]
        mdps = midpoints[i]
        filter = sc<far_bar
        chosen_mdps = mdps[filter]
        passed_midpoints.append(chosen_mdps)

    #condense back into one array
    scores = np.hstack(scores)
    midpoints = np.hstack(passed_midpoints)
    try:
        os.makedirs(f'{savedir}/{start_point}/')
    except FileExistsError:
        None

    plt.plot(all_midpoints[200:-200], scores[200:-200])
    plt.xlabel("Datapoints")
    plt.ylabel("Final metric score")
    plt.savefig(f'{savedir}/{start_point}/eval_timeseries.png', dpi=300)
    plt.close()

    if len(midpoints) == 0:
        print("Found nothing!")
        return None

    midpoint_clusters = clustering(midpoints)
    eval_locations = []
    #print(midpoints)
    for point in midpoint_clusters:
        eval_locations.append(np.where(all_midpoints == point)[0][0])

    matplotlib.rcParams.update({'font.size': 22})

    gps_times = []
    best_scores = []

    output_file = f'{savedir}/event_data.txt'



    for j in range(len(midpoint_clusters)):
        loudest = eval_locations[j]
        #print("loudest", loudest)

        left_edge = 150
        right_edge = 225
        n_points = left_edge+right_edge
        labels = ['background', 'bbh', 'glitch', 'sglf', 'sghf', 'pearson']
        quak_evals_ts = np.linspace(0, n_points*(SEGMENT_OVERLAP/SAMPLE_RATE), n_points)

        #get rid of edge effects
        if loudest < 200 or loudest > len(scores) - 201:
            continue
        try:
            best_score = min(scores[loudest-left_edge:loudest+right_edge])
        except ValueError:
            continue

        fig, axs = plt.subplots(2, 2, figsize=(28, 17))

        # reduce scaled evals of frequency domain correlation
        # reduce scaled evals of frequency domain correlation
        labels = ['background','background', 'bbh','bbh', 'glitch', 'glitch', 'sglf', 'sglf', 'sghf', 'sghf', 'pearson', 'freq corr']
        cols = ['purple', 'blue', 'green', 'salmon', 'goldenrod', 'brown' ]
        for i in range(scaled_evals_.shape[1]):
            #print(scaled_evals_.shape)
            #print(loudest-left_edge, loudest+right_edge, i)
            line_type = "-"
            if i% 2 == 1:
                line_type = "--"
            if i % 2 == 0 or labels[i] in ["pearson", "freq corr"]:
                axs[1, 0].plot(quak_evals_ts*1000, scaled_evals_[loudest-left_edge:loudest+right_edge, i],
                            label = labels[i], c=cols[i//2], linestyle=line_type)
            else:
                axs[1, 0].plot(quak_evals_ts*1000, scaled_evals_[loudest-left_edge:loudest+right_edge, i],
                                c=cols[i//2], linestyle=line_type)
        best_scores.append(best_score)
        axs[1, 0].plot([], [], '-', label="Hanford", c="gray")
        axs[1, 0].plot([], [], '--', label="Livingston", c="gray")
        axs[1, 0].plot(quak_evals_ts*1000, scores[loudest-left_edge:loudest+right_edge]-bias_value, label = 'final metric', c='black')
        axs[1, 0].legend(handlelength=3, fontsize=17)
        axs[1, 0].set_xlabel("Time, (ms)")
        axs[1, 0].set_ylabel("Final Metric Contribution")

        p = all_midpoints[loudest]
        left_edge, right_edge = left_edge * SEGMENT_OVERLAP, right_edge*SEGMENT_OVERLAP
        strain_ts = np.linspace(0, (right_edge+left_edge)/SAMPLE_RATE, right_edge+left_edge)
        axs[0, 0].plot(strain_ts*1000,strain_orig[p-left_edge:p+right_edge, 0], label = 'Hanford', alpha=0.8)
        axs[0, 0].plot(strain_ts*1000, strain_orig[p-left_edge:p+right_edge, 1], label = 'Livingston', alpha=0.8)
        axs[0, 0].set_xlabel('Time, (ms)')
        axs[0, 0].set_ylabel('strain')
        axs[0, 0].legend()
        axs[0, 0].set_title(f'gps time: {p/SAMPLE_RATE:.3f} + {start_point}')
        gps_times.append(p/SAMPLE_RATE + start_point)

        # plot the Q-scans
        q_edge = int(7.5*4096)
        H_strain = data[0][p-left_edge-q_edge:p+right_edge+q_edge]
        L_strain = data[1][p-left_edge-q_edge:p+right_edge+q_edge]
        t0 = H_strain.t0.value
        dt = H_strain.dt.value

        try:
            H_hq = H_strain.q_transform(outseg=(t0+q_edge*dt, t0+q_edge*dt+(left_edge+right_edge)*dt))
            L_hq = L_strain.q_transform(outseg=(t0+q_edge*dt, t0+q_edge*dt+(left_edge+right_edge)*dt))
            f = np.array(H_hq.yindex)
            t = np.array(H_hq.xindex)
            #t=strain_ts *1000
            t -= t[0]

            im_H = axs[0, 1].pcolormesh(t*1000, f, np.array(H_hq).T)
            fig.colorbar(im_H, ax=axs[0, 1], label = "spectral power")
            axs[0, 1].set_yscale("log")
            axs[0, 1].set_xlabel("Time (ms)")
            axs[0, 1].set_ylabel("Freq (Hz)")
            axs[0, 1].set_title("Hanford Q-Transform")

            im_L = axs[1, 1].pcolormesh(t*1000, f, np.array(L_hq).T)
            fig.colorbar(im_L, ax=axs[1, 1], label = "spectral power")
            axs[1, 1].set_yscale("log")
            axs[1, 1].set_xlabel("Time (ms)")
            axs[1, 1].set_ylabel("Freq (Hz)")
            axs[1, 1].set_title("Livingston Q-Transform")

            best_far=""
            if far_search != None:
                far_a, far_b = far_search
                far_a, far_b = np.array(far_a), np.array(far_b)
                #print(far_b, best_score)
                filter = ([elem != None for elem in far_b])
                far_a = far_a[filter][::-1]
                far_b = far_b[filter][::-1]
                loc = np.searchsorted(far_b, best_score)
                best_far = far_a[loc]

            plt.savefig(f'{savedir}/{start_point}/{start_point+p/SAMPLE_RATE:.3f}_{best_far}_{best_score:.2f}{extra}.png', dpi=300, bbox_inches="tight")
            plt.close()

            # Open the output file in write mode
            with open(output_file, "a") as file:
                #write to text file
                file.write(f"{start_point+p/SAMPLE_RATE}\t\t\t\t\t\t{best_score}\t{best_far}\n")
            file.close()
        except:
            pass

    # save data file with detections in the folder
    np.save(f'{savedir}/{start_point}/gps_times.npy', np.array(gps_times))
    np.save(f'{savedir}/{start_point}/best_scores.npy', np.array(best_scores))

def parse_gwtc_catalog(path, mingps=None, maxgps=None):
    gwtc = np.loadtxt(path, delimiter=",", dtype="str")


    pulled_data = np.zeros((gwtc.shape[0]-1, 3))
    for i, elem in enumerate(gwtc[1:]): #first row is just data value description
        pulled_data[i] = [float(elem[4]), float(elem[13]), float(elem[34])]

    if mingps != None:
        assert maxgps != None
        pulled_data = pulled_data[np.logical_and(pulled_data[:, 0]<maxgps, pulled_data[:, 0]>mingps)]
    return pulled_data

def find_segment(gps, segs):
    for seg in segs:
        a, b = seg
        if a < gps and b > gps:
            return seg

def make_metric_vs_far(savepath, bins_path=f"/home/katya.govorkova/gw-anomaly/output/O3bv1/far_bins_k50.npy"):

    search_vals = np.logspace(-3, np.log10(100), 200) # in number per month
    search_vals *= 30*24*3600

    metric_vals = []
    far_bins = np.load(bins_path)
    for elem in search_vals:
        metric_vals.append(far_to_metric(elem, far_bins))

    plt.xscale("log")
    plt.plot(1/search_vals, metric_vals)

    vbars = [3600, 24*3600, 7*24*3600, 30*24*3600]
    labels = ["hour", "day", "week", "month"]
    for i, val in enumerate(vbars):
        plt.axvline(x=1/val, label=labels[i], alpha=0.9**(len(labels)-i), c="black")
    plt.legend()
    plt.xlabel("FAR, Hz")
    plt.ylabel("corresponding metric values")
    plt.savefig(f"{savepath}/FAR_vs_metric.pdf", dpi=300)
    return 1/search_vals, metric_vals

def main(args):
    lst = os.listdir('/scratch/florent.robinet/BurstBenchmark/')
    lst = np.array(lst)
    lst = [i[5:] for i in lst if "V1" not in i]
    lst_noduplicates = [*set(lst)]

    trained_path = MODELS_LOCATION

    savedir = args.outdir
    try:
        os.makedirs(savedir)
        print("made dir", savedir)
    except FileExistsError:
        None
    far_a, far_b = make_metric_vs_far(savedir)

    from tqdm import tqdm

    for file in tqdm(lst_noduplicates, desc="Processing files"):
        data, A = whiten_bandpass_resample(file, savedir, shift=None)

        #A is the start GPS time for the sample

        # data is [strain_H1, strain_L1], where each is downsampled, whitened, and bandpassed
        get_evals(data, trained_path, savedir, int(A), far_search=[far_a, far_b])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--valid-segments', type=str, default=None,
                        help='File with valid segments')
    parser.add_argument('--outdir', type=str, default=None,
                        help='output directory')
    args = parser.parse_args()
    main(args)