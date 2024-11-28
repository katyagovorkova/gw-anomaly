import os
import numpy as np
import argparse
import torch
from gwpy.timeseries import TimeSeries
from astropy import units as u
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = False
from models import LinearModel, GwakClassifier
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from evaluate_data import full_evaluation
import json
import matplotlib
from torch.nn.functional import conv1d
from scipy.stats import pearsonr
import pickle
from config import (
    CHANNEL,
    GPU_NAME,
    SEGMENT_OVERLAP,
    SAMPLE_RATE,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    FACTORS_NOT_USED_FOR_FM,
    MODELS_LOCATION,
    SEG_NUM_TIMESTEPS,
    CLASS_ORDER
    )
import torch.nn as nn
class BasedModel(nn.Module):
    def __init__(self):
        super(BasedModel, self).__init__()

        self.layer1 = nn.Linear(3, 1)
        self.layer2_1 = nn.Linear(1, 1)
        self.layer2_2 = nn.Linear(1, 1)
        self.layer2_3 = nn.Linear(1, 1)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.activation(self.layer1(x[:, :3]))
        x2_1 = self.activation(self.layer2_1(x[:, 3:4]))
        x2_2 = self.activation(self.layer2_1(x[:, 4:5]))
        x2_3 = self.activation(self.layer2_1(x[:, 5:6]))
        return x1 * x2_1 * x2_2 * x2_3

def extract(gwak_values):
    result = np.zeros((gwak_values.shape[0], 3))
    for i, pair in enumerate([[3, 4], [9, 10], [12, 13]]):
        a, b = pair
        ratio_a = (np.abs(gwak_values[:, a]) + 2) / (np.abs(gwak_values[:, b]) + 2)
        ratio_b = (np.abs(gwak_values[:, b]) + 2) / (np.abs(gwak_values[:, a]) + 2)

        ratio = np.maximum(ratio_a, ratio_b)
        result[:, i] = ratio
    return result

def compute_signal_strength_chop_sep(x, y):
    psd0 = welch(x)[1]
    psd1 = welch(y)[1]
    HLS = np.log(np.sum(psd0))
    LLS = np.log(np.sum(psd1))
    return HLS, LLS

def shifted_pearson(H, L, H_start, H_end, maxshift=int(10*4096/1000)):
    # works for one window at a time
    Hs = H[H_start:H_end]
    minval = 1
    for shift in range(-maxshift, maxshift):
        Ls = L[H_start+shift:H_end+shift]

        p = pearsonr(Hs, Ls)[0]
        if p < minval:
            minval = p
            shift_idx = shift

    return minval, shift_idx

def parse_strain(x):
    # take strain, compute the long sig strenght & pearson
    # split it up, do the same thing for short
    long_pearson, shift_idx = shifted_pearson(x[0], x[1], 50, len(x[0])-50)
    HSS, LSS = compute_signal_strength_chop_sep(x[0, 50:-50], x[1, 50+shift_idx:len(x[0])-50+shift_idx])
    return long_pearson, HSS, LSS

from helper_functions import far_to_metric, compute_fars, load_gwak_models, joint_heuristic_test, combine_freqcorr
DEVICE = torch.device(GPU_NAME)
from scipy.signal import welch
heuristics_tests = True

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

def sig_prob_function(evals, scale=40):
    sigmoid = lambda x: 1/(1+np.exp(-x))
    return 1-(sigmoid(scale * (evals-0.5)))

def get_far(score, sort_eval):
    ind = np.searchsorted(sort_eval, score)
    if ind == len(sort_eval):
        ind -= 1
    #N = len(sort_eval)
    units = 10000*3.15e7
    return ind/units

def make_eval_chunks(a, b, dur):
    '''
    Split up into one-hour chunks to normalize the whitening duration
    a, b - ints
    A, B - strings

    output - only care about the strings
    '''
    n_full_chunks = (b-a)//dur

    out = []
    for n in range(1, n_full_chunks+1):
        out.append([str(a+(n-1)*dur), str(a+n*dur)])

    #ending chunk, but still make it one hour
    out.append([str(b-dur), str(b)])
    return out

def event_clustering(indices, scores, spacing, device):
    '''
    Group the evaluations into events, i.e. treat a consecutive sequence
    of low anomaly score as a single event
    '''

    clustered = []
    idxs = indices.detach().cpu().numpy()
    cluster = []
    for i, elem in enumerate(idxs):
        # to move onto next cluster
        if i != 0:
            dist = elem - idxs[i-1]
            if dist > spacing:
                #make a new cluster
                clustered.append(cluster)
                cluster = [] # and initiate a new one
        cluster.append(elem)
    clustered.append(cluster) # last one isn't captured, since we haven't moved on
    final_points = []
    for cluster in clustered:
        # take the one with the lowest score (most significant)
        bestscore = 10
        bestval = None
        for elem in cluster:
            if scores[elem] < bestscore:
                bestscore = scores[elem]
                bestval = elem
        final_points.append(bestval)
    return torch.from_numpy(np.array(final_points)).int().to(device)

def extract_chunks(strain_data, timeslide_num, important_points, device,
                    roll_amount = SEG_NUM_TIMESTEPS, window_size=1024):
    '''
    Important points are indicies into thestrain_data
    '''
    L_shift = timeslide_num*roll_amount
    timeslide_len = strain_data.shape[1]
    edge_check_passed = []
    fill_strains = np.zeros((len(important_points), 2, window_size*2))
    for idx, point in enumerate(important_points):
        # check that the point is not on the edge
        edge_check_passed.append(not(point < window_size * 2 or timeslide_len - point < window_size*2))
        if not(point < window_size * 2 or timeslide_len - point < window_size*2):
            H_selection = strain_data[0, point-window_size:point+window_size]

            # if the livingston points overflow, the modulo should bring them
            # into the right location. also data is clipped //1000 * 1000
            # which is divisible by 200, so it should work
            L_start = (point-window_size+L_shift) % timeslide_len
            L_end = (point+window_size+L_shift) % timeslide_len

            L_selection = strain_data[1, L_start:L_end]

            fill_strains[idx, 0, :] = H_selection
            fill_strains[idx, 1, :] = L_selection

    return fill_strains, edge_check_passed

def rwb(
        start_point,
        end_point,
        sample_rate=SAMPLE_RATE,
        bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH,
        shift=None):

    try:
        start_point, end_point = int(start_point)-20, int(end_point)-20
        strainL1 = TimeSeries.fetch_open_data(f'L1', start_point, end_point)
        strainH1 = TimeSeries.fetch_open_data(f'H1', start_point, end_point)

        t0 = int(strainL1.t0 / u.s)

        # Whiten, bandpass, and resample
        strainL1 = strainL1.resample(sample_rate)
        strainL1 = strainL1.whiten().bandpass(bandpass_low, bandpass_high)

        strainH1 = strainH1.resample(sample_rate)
        strainH1 = strainH1.whiten().bandpass(bandpass_low, bandpass_high)

        return [strainH1, strainL1]
    except:
        return None, None

def rbw(
        start_point,
        end_point,
        sample_rate=SAMPLE_RATE,
        bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH,
        shift=None):

    try:
        start_point, end_point = int(start_point)-20, int(end_point)-20
        strainL1 = TimeSeries.fetch_open_data(f'L1', start_point, end_point)
        strainH1 = TimeSeries.fetch_open_data(f'H1', start_point, end_point)

        t0 = int(strainL1.t0 / u.s)

        # Whiten, bandpass, and resample
        strainL1 = strainL1.resample(sample_rate)
        strainL1 = strainL1.bandpass(bandpass_low, bandpass_high).whiten()

        strainH1 = strainH1.resample(sample_rate)
        strainH1 = strainH1.bandpass(bandpass_low, bandpass_high).whiten()

        return [strainH1, strainL1]
    except:
        return None, None


def get_evals(data_, model_path, savedir, start_point, gwpy_timeseries):

    model_path = "/home/katya.govorkova/gwak-paper-final-models/trained/model_heuristic.h5"
    model_heuristic = BasedModel().to(DEVICE)
    model_heuristic.load_state_dict(torch.load(model_path, map_location=DEVICE))

    eval_at_once_len = int(3600)
    N_one_hour_splits = int(data_.shape[1]//(eval_at_once_len*SAMPLE_RATE) + 1)

    for hour_split in range(N_one_hour_splits):
        start = int(hour_split*SAMPLE_RATE*eval_at_once_len)
        end = int(min(data_.shape[1], (hour_split+1)*SAMPLE_RATE*eval_at_once_len))
        # print(start, end)
        if end - 10 < start:
            return None
        data = data_[:, start:end]

        model_path = ["output/O3av2/trained/models/bbh.pt",
                      "output/O3av2/trained/models/sglf.pt",
                      "output/O3av2/trained/models/sghf.pt",
                      "output/O3av2/trained/models/background.pt",
                      "output/O3av2/trained/models/glitches.pt"]

        models_path = [os.path.join(MODELS_LOCATION, os.path.basename(f)) for f in model_path]
        gwak_models = load_gwak_models(models_path, DEVICE, GPU_NAME)
        orig_kernel = 50
        kernel_len = int(orig_kernel * 5/SEGMENT_OVERLAP)
        kernel = torch.ones((1, kernel_len)).float().to(DEVICE)/kernel_len
        kernel = kernel[None, :, :]
        heuristics_tests = True
        norm_factors = np.load(f"/home/katya.govorkova/gwak-paper-final-models/trained/norm_factor_params.npy")

        fm_model_path = ("/home/katya.govorkova/gwak-paper-final-models/trained/fm_model.pt")
        fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
        fm_model.load_state_dict(torch.load(
            fm_model_path, map_location=GPU_NAME))

        linear_weights = fm_model.layer.weight.detach()#.cpu().numpy()
        bias_value = fm_model.layer.bias.detach()#.cpu().numpy()
        linear_weights[:, -2] += linear_weights[:, -1]
        linear_weights = linear_weights[:, :-1]
        norm_factors = norm_factors[:, :-1]

        mean_norm = torch.from_numpy(norm_factors[0]).to(DEVICE)#[:-1]
        std_norm = torch.from_numpy(norm_factors[1]).to(DEVICE)#[:-1]

        final_values, midpoints, original, recreated = full_evaluation(
                        data[None, :, :], models_path, DEVICE,
                        return_midpoints=True, return_recreations=True,
                        loaded_models=gwak_models, grad_flag=False)

        final_values = final_values[0]

        # Set the threshold here
        FAR_2days = 0 # lowest FAR bin we want to worry about

        # Inference to save scores (final metric) and scaled_evals (GWAK space * weights unsummed)
        final_values_slx = (final_values - mean_norm)/std_norm
        scaled_evals = torch.multiply(final_values_slx, linear_weights[None, :])[0, :]
        scores = (scaled_evals.sum(axis=1) + bias_value)[:, None]
        scaled_evals = conv1d(scaled_evals.transpose(0, 1).float()[:, None, :],
            kernel, padding="same").transpose(0, 1)[0].transpose(0, 1)
        smoothed_scores = conv1d(scores.transpose(0, 1).float()[:, None, :],
            kernel, padding="same").transpose(0, 1)[0].transpose(0, 1)
        indices = torch.where(smoothed_scores < FAR_2days)[0]

        if len(indices) == 0: continue # Didn't find anything

        indices = event_clustering(indices, smoothed_scores, 5*SAMPLE_RATE/SEGMENT_OVERLAP, DEVICE) # 5 seconds
        filtered_final_score = smoothed_scores.index_select(0, indices)
        filtered_final_scaled_evals = scaled_evals.index_select(0, indices)

        indices = indices.detach().cpu().numpy()
        # extract important "events" with indices
        timeslide_chunks, edge_check_filter = extract_chunks(data, 0, # 0 - timeslide number 0 (no shifting happening)
                                            midpoints[indices],
                                            DEVICE, window_size=1024) # 0.25 seconds on either side
                                                                    # so it should come out to desired 0.5

        filtered_final_scaled_evals = filtered_final_scaled_evals.detach().cpu().numpy()
        filtered_final_score = filtered_final_score.detach().cpu().numpy()

        filtered_final_scaled_evals = filtered_final_scaled_evals[edge_check_filter]
        filtered_final_score = filtered_final_score[edge_check_filter]
        timeslide_chunks = timeslide_chunks[edge_check_filter]
        indices = indices[edge_check_filter]

        filtered_timeslide_chunks = timeslide_chunks

        heuristics_tests = True
        if heuristics_tests:
            N_initial = len(filtered_final_score)
            passed_heuristics = []
            gwak_filtered = extract(filtered_final_scaled_evals)
            for i, strain_segment in enumerate(timeslide_chunks):
                strain_feats = parse_strain(strain_segment)
                together = np.concatenate([strain_feats, gwak_filtered[i]])
                res = model_heuristic(torch.from_numpy(together[None, :]).float().to(DEVICE)).item()

                res_sigmoid = sig_prob_function(res)
                final_final = res_sigmoid * filtered_final_score[i]
                filtered_final_score[i] *= res_sigmoid
                passed_heuristics.append(final_final[0] < -0.5) #[0] since it was saving arrays(arrays)


            filtered_final_scaled_evals = filtered_final_scaled_evals[passed_heuristics]
            filtered_final_score = filtered_final_score[passed_heuristics]
            filtered_timeslide_chunks = timeslide_chunks[passed_heuristics]
            indices = indices[passed_heuristics]

            print(f"Fraction removed by heuristics test {N_initial -len(filtered_final_score)}/{N_initial}")
        # rename them for less confusion, easier typing
        gwak_values = filtered_final_scaled_evals
        fm_scores = filtered_final_score
        strain_chunks = filtered_timeslide_chunks

        if strain_chunks.shape[0] == 0: continue
        # plotting all these significant events
        n_points = strain_chunks.shape[2]

        scaled_evals = scaled_evals.cpu().numpy()
        scaled_evals = combine_freqcorr(scaled_evals)
        bias_value = bias_value.cpu().numpy()
        smoothed_scores = smoothed_scores.cpu().numpy()

        for j in range(len(gwak_values)):
            fig1, axs1 = plt.subplots(4, 1, figsize=(10, 12) ) #, sharex=True)  # Strain and GWAK values (shared x-axis)
            # fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)  # Hanford and Livingston Q-transforms (shared y-axis)

            loudest = indices[j]
            left_edge = 1024 // SEGMENT_OVERLAP
            right_edge = 1024 // SEGMENT_OVERLAP
            quak_evals_ts = np.linspace(0, (left_edge + right_edge) * SEGMENT_OVERLAP / SAMPLE_RATE, left_edge + right_edge)

            labels = ['Background', 'Background', 'BBH', 'BBH', 'Glitch', 'Glitch', 'SG (64-512 Hz)', 'SG (64-512 Hz)', 'SG (512-1024 Hz)', 'SG (512-1024 Hz)', 'Frequency correlation']
            cols = [
                "#f4a3c1",  # Background (Soft Pink)
                "#ffd700",  # BBH (Yellow - Gold)
                "#2a9d8f",  # Glitch (Emerald Green)
                "#708090",  # SGLF (Light Slate Gray)
                "#00bfff",  # SGHF (Deep Sky Blue)
                "#cd5c5c",  # Freq Corr (Indian Red)
                "#006400",  # Final Metric (Dark Green)
                "#daa520",  # Hanford (Goldenrod)
                "#ff6347",  # Livingston (Tomato)
            ]

            # Strain plot
            strain_ts = np.linspace(0, len(strain_chunks[j, 0, :]) / SAMPLE_RATE, len(strain_chunks[j, 0, :]))
            axs1[2].plot(strain_ts, gwpy_timeseries[0][midpoints[loudest]-1024:midpoints[loudest]+1024], label='Hanford', alpha=0.8, c="#6c5b7b")
            axs1[2].plot(strain_ts, gwpy_timeseries[1][midpoints[loudest]-1024:midpoints[loudest]+1024], label='Livingston', alpha=0.8, c="#f29e4c")
            axs1[2].set_ylabel('Strain', fontsize=14)
            axs1[2].legend()
            # axs1[0].set_title(f'GPS time: {start_point + midpoints[loudest] / SAMPLE_RATE + hour_split * eval_at_once_len:.1f}')
            # Remove x-axis ticks
            axs1[2].tick_params(axis='x', which='both', bottom=False, top=False)
            axs1[2].set_xticklabels([])

            # GWAK values plot
            for i in range(scaled_evals.shape[1]):
                line_type = "-" if i % 2 == 0 else "--"
                axs1[3].plot(
                    1000 * quak_evals_ts,
                    scaled_evals[loudest - left_edge:loudest + right_edge, i],
                    label=labels[i] if i % 2 == 0 or labels[i] in ["Frequency correlation"] else None,
                    c=cols[i // 2],
                    linestyle=line_type
                )
            axs1[3].plot(
                1000 * quak_evals_ts,
                smoothed_scores[loudest - left_edge:loudest + right_edge] - bias_value,
                label='Final metric',
                c='black'
            )
            axs1[3].set_xlabel("Time (ms)", fontsize=14)
            axs1[3].set_ylabel("Final metric contributions", fontsize=14)
            axs1[3].legend()

            # Define a custom colormap with pink in the middle
            custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#1f77b4", "#f4a3c1", "#ffd700"], N=256)

            # Define shared color scale range and normalization (pink in the middle)
            vmin, vcenter, vmax = 0, 12.5, 25  # Pink at 12.5, scale from 0 to 25
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

            # Hanford and Livingston Q-Transforms
            p = midpoints[loudest]
            left_edge = 1024
            right_edge = 1024
            q_edge = int(7.5 * 4096)

            H_strain = gwpy_timeseries[0][
                p - left_edge - q_edge + eval_at_once_len * hour_split * SAMPLE_RATE:
                p + right_edge + q_edge + eval_at_once_len * hour_split * SAMPLE_RATE
            ]
            L_strain = gwpy_timeseries[1][
                p - left_edge - q_edge + eval_at_once_len * hour_split * SAMPLE_RATE:
                p + right_edge + q_edge + eval_at_once_len * hour_split * SAMPLE_RATE
            ]

            t0 = H_strain.t0.value
            dt = H_strain.dt.value
            H_hq = H_strain.q_transform(outseg=(t0 + q_edge * dt, t0 + q_edge * dt + (left_edge + right_edge) * dt), whiten=False)
            L_hq = L_strain.q_transform(outseg=(t0 + q_edge * dt, t0 + q_edge * dt + (left_edge + right_edge) * dt), whiten=False)

            f = np.array(H_hq.yindex)
            t = np.array(H_hq.xindex)
            t -= t[0]

            # Create the Q-Transform plots
            im_H = axs1[0].pcolormesh(
                t * 1000,
                f,
                np.array(H_hq).T,
                cmap=custom_cmap,
                norm=norm,
                shading="auto"
            )
            axs1[0].set_yscale("log")
            axs1[0].set_ylabel("Frequency (Hz)", fontsize=14)
            # axs1[1].set_title("Hanford Q-Transform", fontsize=14)
            axs1[1].tick_params(axis='x', which='both', bottom=False, top=False)
            axs1[0].tick_params(axis='x', which='both', bottom=False, top=False)
            axs1[1].set_xticklabels([])
            axs1[0].set_xticklabels([])


            im_L = axs1[1].pcolormesh(
                t * 1000,
                f,
                np.array(L_hq).T,
                cmap=custom_cmap,
                norm=norm,
                shading="auto"
            )
            axs1[1].set_yscale("log")
            # axs1[1].set_xlabel("Time (ms)", fontsize=12)
            axs1[1].set_ylabel("Frequency (Hz)", fontsize=14)
            # axs1[1].set_title("Livingston Q-Transform", fontsize=14)

            # Add shared colorbar
            cbar = fig1.colorbar(
                im_H,
                ax=axs1[0],
                location="top",
                pad=0.05,
                # shrink=0.9,
                aspect=30
            )
            cbar.set_label("Spectral Power", fontsize=12)

            # Adjust layout
            fig1.tight_layout()



            # Save figures
            base = f'{savedir}/{start_point + p / SAMPLE_RATE:.3f}_0_{fm_scores[j][0]:.2f}'
            fig1.savefig(f'{base}.png', dpi=300, bbox_inches="tight")
            # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            # fig2.savefig(f'{base}_q_transforms.png', dpi=300, bbox_inches="tight")
            plt.close(fig1)
            # plt.close(fig2)

            # Save data
            np.savez(
                f"{base}.npz",
                {
                    "strain": [strain_ts, strain_chunks[j]],
                    "gwak_values": [1000 * quak_evals_ts, smoothed_scores[loudest - left_edge:loudest + right_edge] - bias_value],
                    "H_qtransform": [t * 1000, f, np.array(H_hq).T],
                    "L_qtransform": [t * 1000, f, np.array(L_hq).T],
                }
            )
            print('SAVED', base)

def main(args):
    try:
        os.makedirs(args.savedir)
    except FileExistsError:
        None

    valid_segments = np.load("/home/katya.govorkova/gw_anomaly/output/O3a_intersections.npy")
    trained_path = "/home/katya.govorkova/gwak-paper-final-models/"
    run_short_test = False
    if run_short_test:
        A = 1241104246.7490
        B = A + 3600

        H, L = rwb(A, B)
        data = np.vstack([np.array(H.data), np.array(L.data)])

        get_evals(data, trained_path, args.savedir, int(A), [H, L])
        assert 0

    f_segments = 'paperO3a/done_O3a_segments.npy'
    if not os.path.exists(f_segments):
        np.save(f_segments, np.array([0]))

    anomaly_start_times = [
        # (1239155734.182, -1.1),
        # (1240878400.307, -2.31),
        (1241104246.749, -1.8),
        # (1241624696.55, -1.03),
        # (1242442957.423, -2.69),
        # (1242459847.413, -1.12),
        # (1242827473.37, -1.13),
        (1243305662.931, -6.0),
        # (1245998824.997, -1.03),
        # (1246417246.823, -1.21),
        # (1246487209.308, -3.59),
        # (1247281292.53, -1.01),
        # (1248280604.554, -1.11),
        # (1249035984.212, -1.35),
        # (1249635282.359, -1.49),
        # (1250981809.437, -1.4),
        # (1251009253.724, -4.76),
        # (1252679441.276, -1.09),
        # (1252833818.202, -1.11),
        # (1253638396.336, -1.4),
        # (1257416710.328, -1.21),
        # (1260164266.18, -1.1),
        # (1260358297.149, -1.01),
        # (1260825537.025, -1.75),
        # (1261020945.101, -1.03),
        # (1262203609.392, -3.98),
        # (1263013357.045, -6.49),
        # (1264316106.385, -1.55),
        # (1264683185.946, -1.04),
        # (1266473981.889, -1.02),
        # (1267610448.007, -1.92),
        # (1267610483.017, -6.13),
        # (1267617688.034, -5.61),
        # (1267878076.354, -5.74),
        # (1269242528.39, -2.0)
        (1251009253.724, -5),
        (1263013367.055, -5)
        ]


    for A, _ in anomaly_start_times:
            B = A + 3600

            print("Working on", A, B)
            H, L = rwb(A, B)
            H_rbw, L_rbw = rbw(A, B)

            if H is None and L is None: continue
            data = np.vstack([np.array(H.data), np.array(L.data)])
            if data.shape[-1] < 1e5: return None
            get_evals(data, trained_path, args.savedir, int(A), [H_rbw, L_rbw])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--savedir', type=str, default='paperO3a',
                        help='File with valid segments')

    args = parser.parse_args()
    main(args)