import pickle
import time
import os
import numpy as np
import argparse
import torch
from gwpy.timeseries import TimeSeries
from gwpy.segments import DataQualityDict
from astropy import units as u
import matplotlib.pyplot as plt
from models import LinearModel, GwakClassifier
from evaluate_data import full_evaluation
import json
import matplotlib
from torch.nn.functional import conv1d
from scipy.stats import pearsonr
import torch.nn as nn
from scipy.signal import welch
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
    CLASS_ORDER,
    FM_LOCATION
    )
from helper_functions import (
    far_to_metric,
    compute_fars,
    load_gwak_models,
    joint_heuristic_test,
    combine_freqcorr
    )

CHANNEL = 'DCS-CALIB_STRAIN_CLEAN_C01'
FRAME_TYPE = 'HOFT_C01'
STATE_FLAG = 'DCS-ANALYSIS_READY_C01:1'
CUDA_LAUNCH_BLOCKING = 1
DEVICE = torch.device(GPU_NAME)
heuristics_tests = True


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
        # x2_1 = self.activation(self.layer2_1(x[:, 3:4]))
        # x2_2 = self.activation(self.layer2_2(x[:, 4:5]))
        # x2_3 = self.activation(self.layer2_3(x[:, 5:6]))
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
        #minval = min(pearsonr(Hs, Ls)[0], minval)
        p = pearsonr(Hs, Ls)[0]
        if p < minval:
            minval = p
            shift_idx = shift

    return minval, shift_idx

def parse_strain(x):
    # take strain, compute the long sig strenght & pearson
    # split it up, do the same thing for short
    long_pearson, shift_idx = shifted_pearson(x[0], x[1], 50, len(x[0])-50)
    #long_sig_strength = compute_signal_strength_chop(x[0, 50:-50], x[1, 50+shift_idx:len(x[0])-50+shift_idx] )
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
    #sigmoid = lambda x: 1/(1+np.exp(-(x-0.3)))
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


def resample_bandpass_whiten(
        start_point,
        end_point,
        sample_rate=SAMPLE_RATE,
        bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH,
        shift=None):

    device = torch.device(GPU_NAME)

    start_point, end_point = int(start_point)+10, int(end_point)-10
    strainL1_0 = TimeSeries.get(f'L1:{CHANNEL}', start_point, end_point)
    strainH1_0 = TimeSeries.get(f'H1:{CHANNEL}', start_point, end_point)

    strainL1_rbw = strainL1_0.resample(sample_rate)
    strainL1_rbw = strainL1_rbw.bandpass(bandpass_low, bandpass_high).whiten()
    strainH1_rbw = strainH1_0.resample(sample_rate)
    strainH1_rbw = strainH1_rbw.bandpass(bandpass_low, bandpass_high).whiten()

    strainL1 = strainL1_0.resample(sample_rate)
    strainL1 = strainL1.whiten().bandpass(bandpass_low, bandpass_high)
    strainH1 = strainH1_0.resample(sample_rate)
    strainH1 = strainH1.whiten().bandpass(bandpass_low, bandpass_high)

    return strainH1, strainL1, strainH1_rbw, strainL1_rbw


def read_segments_from_file(filename):
    segments = []
    with open(filename, 'r') as file:
        for line in file:
            start, end = map(float, line.split())
            segments.append((start, end))
    return segments


def check_segment_overlap(detection_point, segments):
    for (start, end) in segments:
        if not (detection_point < start or detection_point > end):
            return True  # Overlap found
    return False  # No overlap found


def get_evals(data_, model_path, savedir, start_point,
              gwpy_timeseries,  data_rbw=None, detection_point=None, manual_eval_times=None):

    heur_model_path = f"{FM_LOCATION}/model_heuristic.h5"
    model_heuristic = BasedModel().to(DEVICE)
    model_heuristic.load_state_dict(torch.load(heur_model_path, map_location=DEVICE))


    #model_path = '/n/home00/emoreno/katya_LITERALLY/gw_anomaly/ryan/model.h5'
    #model_path = "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/plots/model.h5"
    # heur_model_path = "/home/ryan.raikman/ss24/gw-anomaly/gw_anomaly/output/plots/model.h5"
    # model_heuristic = BasedModel().to(DEVICE)
    # model_heuristic.load_state_dict(torch.load(heur_model_path, map_location=DEVICE))


    # split the data into 1-hour chunks to fit in memory best
    eval_at_once_len = int(3600)
    N_one_hour_splits = int(data_.shape[1]//(eval_at_once_len*SAMPLE_RATE) + 1)
    print("N splits:", N_one_hour_splits)

    for hour_split in range(N_one_hour_splits):
        start = int(hour_split*SAMPLE_RATE*eval_at_once_len)
        end = int(min(data_.shape[1], (hour_split+1)*SAMPLE_RATE*eval_at_once_len))
        print(start, end)
        if end - 10 < start:
            return None
        data = data_[:, start:end]

        model_paths = [f"{MODELS_LOCATION}/bbh.pt",
                       f"{MODELS_LOCATION}/sglf.pt",
                       f"{MODELS_LOCATION}/sghf.pt",
                       f"{MODELS_LOCATION}/background.pt",
                       f"{MODELS_LOCATION}/glitches.pt"]

        gwak_models = load_gwak_models(model_paths, DEVICE, GPU_NAME)

        # norm_factors = np.load(f"/home/ryan.raikman/ss24/gw-anomaly/gw_anomaly/output/O3av0/trained/norm_factor_params.npy")
        # fm_model_path = ("/home/ryan.raikman/ss24/gw-anomaly/gw_anomaly/output/O3av0/trained/fm_model.pt")
        norm_factors = np.load(f"/home/katya.govorkova/gwak-paper-final-models/trained/norm_factor_params.npy")
        fm_model_path = ("/home/katya.govorkova/gwak-paper-final-models/trained/fm_model.pt")

        orig_kernel = 50
        kernel_len = int(orig_kernel * 5/SEGMENT_OVERLAP)
        kernel = torch.ones((1, kernel_len)).float().to(DEVICE)/kernel_len
        kernel = kernel[None, :, :]
        heuristics_tests = True

        if 0:
            fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)-1).to(DEVICE)
            fm_model.load_state_dict(torch.load(
                fm_model_path, map_location=GPU_NAME))

            linear_weights = fm_model.layer.weight.detach()#.cpu().numpy()
            bias_value = fm_model.layer.bias.detach()#.cpu().numpy()
            # linear_weights[:, -2] += linear_weights[:, -1]
            # linear_weights = linear_weights[:, :-1]
            # norm_factors = norm_factors[:, :-1]
        else:
            fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
            fm_model.load_state_dict(torch.load(
                fm_model_path, map_location=GPU_NAME))

            linear_weights = fm_model.layer.weight.detach()#.cpu().numpy()
            bias_value = fm_model.layer.bias.detach()#.cpu().numpy()
            linear_weights[:, -2] += linear_weights[:, -1]
            linear_weights = linear_weights[:, :-1]
            norm_factors = norm_factors[:, :-1]

        orig_kernel = 50
        kernel_len = int(orig_kernel * 5/SEGMENT_OVERLAP)
        kernel = torch.ones((1, kernel_len)).float().to(DEVICE)/kernel_len
        kernel = kernel[None, :, :]
        heuristics_tests = True



        mean_norm = torch.from_numpy(norm_factors[0]).to(DEVICE)#[:-1]
        std_norm = torch.from_numpy(norm_factors[1]).to(DEVICE)#[:-1]

        #final_values, midpoints = full_evaluation(
        #                data[None, :, :], models_path, DEVICE,
        #                return_midpoints=True, loaded_models=gwak_models, grad_flag=False)
        final_values, midpoints, original, recreated = full_evaluation(
                        data[None, :, :], model_paths, DEVICE,
                        return_midpoints=True, return_recreations=True,
                        loaded_models=gwak_models, grad_flag=False)

        print(335, final_values.shape)
        print(336, midpoints.shape)

        final_values = final_values[0]

        # Set the threshold here
        FAR_2days = -1 # lowest FAR bin we want to worry about

        # Inference to save scores (final metric) and scaled_evals (GWAK space * weights unsummed)
        final_values_slx = (final_values - mean_norm)/std_norm
        scaled_evals = torch.multiply(final_values_slx, linear_weights[None, :])[0, :]
        #print("scaled_evals", scaled_evals.shape)
        scores = (scaled_evals.sum(axis=1) + bias_value)[:, None]
        scaled_evals = conv1d(scaled_evals.transpose(0, 1).float()[:, None, :],
            kernel, padding="same").transpose(0, 1)[0].transpose(0, 1)
        smoothed_scores = conv1d(scores.transpose(0, 1).float()[:, None, :],
            kernel, padding="same").transpose(0, 1)[0].transpose(0, 1)
        indices = torch.where(smoothed_scores < FAR_2days)[0]


        #here, add manual ones
        # bit of code from below that gives the gps time of the event
        #loudest = indices[j]
        #midpoints[loudest]/SAMPLE_RATE + hour_split*eval_at_once_len
        #second half isn't important
        manual_eval_times = torch.tensor(manual_eval_times).to(indices.device)
        if manual_eval_times[0] > start_point:
            manual_eval_times -= start_point

        manual_eval_indices = torch.zeros_like(manual_eval_times)

        for i, eval_time in enumerate(manual_eval_times):
            eval_time = eval_time * SAMPLE_RATE
            insert_location = torch.searchsorted(torch.from_numpy(midpoints).to(eval_time.device), eval_time)
            manual_eval_indices[i] = insert_location



        if len(indices) == 0: continue # Didn't find anything

        indices = event_clustering(indices, smoothed_scores, 5*SAMPLE_RATE/SEGMENT_OVERLAP, DEVICE) # 5 seconds
        indices = torch.cat([indices, manual_eval_indices])
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
                print(433, "together", together, filtered_final_score[i])
                res = model_heuristic(torch.from_numpy(together[None, :]).float().to(DEVICE)).item()
                #passed_heuristics.append(res<0.46)
                #res -= 0.1

                res_sigmoid = sig_prob_function(res)
                #print(res, res_sigmoid, filtered_final_score[i])
                final_final = res_sigmoid * filtered_final_score[i]
                #print(res, res_sigmoid, filtered_final_score[i])
                filtered_final_score[i] *= res_sigmoid
                print(res, res_sigmoid, filtered_final_score[i])
                print('final_final', final_final[0])
                # passed_heuristics.append(final_final[0] < -1.75) #[0] since it was saving arrays(arrays)
                passed_heuristics.append(final_final[0] < -0.46) #[0] since it was saving arrays(arrays)


            filtered_final_scaled_evals = filtered_final_scaled_evals[passed_heuristics]
            filtered_final_score = filtered_final_score[passed_heuristics]
            filtered_timeslide_chunks = timeslide_chunks[passed_heuristics]
            indices = indices[passed_heuristics]

            print(f"Fraction removed by heuristics test {N_initial -len(filtered_final_score)}/{N_initial}")
        # rename them for less confusion, easier typing
        gwak_values = filtered_final_scaled_evals
        fm_scores = filtered_final_score
        strain_chunks = filtered_timeslide_chunks

        if strain_chunks.shape[0] == 0:
            print('EMPTY')
            continue
        # plotting all these significant events
        n_points = strain_chunks.shape[2]

        scaled_evals = scaled_evals.cpu().numpy()
        scaled_evals = combine_freqcorr(scaled_evals)
        bias_value = bias_value.cpu().numpy()
        smoothed_scores = smoothed_scores.cpu().numpy()


        # j = 0
        for j in range(len(gwak_values)):
            indiv_fig = np.empty((2, 2), dtype=object)
            indiv_axs = np.empty((2, 2), dtype=object)
            for a in range(2):
                #indiv_fig[a] = dict()
                for b in range(2):
                    temp_fig, temp_axs = plt.subplots(1, 1, figsize=(8, 5))
                    #print(a, b, temp_fig)
                    indiv_fig[a, b] = temp_fig
                    indiv_axs[a, b] = temp_axs

            #assert 0

            #fig, axs = plt.subplots(2, 2, figsize=(28, 17))
            loudest = indices[j]
            left_edge = 1024 //SEGMENT_OVERLAP
            right_edge = 1024 // SEGMENT_OVERLAP
            quak_evals_ts = np.linspace(0, (left_edge+right_edge)*SEGMENT_OVERLAP/SAMPLE_RATE  , left_edge+right_edge)
            labels = ['background','background', 'bbh','bbh', 'glitch', 'glitch', 'sglf', 'sglf', 'sghf', 'sghf', 'freq corr']
            cols = ['purple', 'blue', 'green', 'salmon', 'goldenrod', 'brown' ]
            for i in range(scaled_evals.shape[1]):
                line_type = "-"
                if i% 2 == 1:
                    line_type = "--"
                if i % 2 == 0 or labels[i] in ["freq corr"]:

                    indiv_axs[1, 0].plot(1000*quak_evals_ts, scaled_evals[loudest-left_edge:loudest+right_edge, i],
                                label = labels[i], c=cols[i//2], linestyle=line_type)

                else:
                    indiv_axs[1, 0].plot(1000*quak_evals_ts, scaled_evals[loudest-left_edge:loudest+right_edge, i],
                                    c=cols[i//2], linestyle=line_type)


            indiv_axs[1, 0].plot(1000*quak_evals_ts, smoothed_scores[loudest-left_edge:loudest+right_edge]-bias_value, label = 'final metric', c='black')
            indiv_axs[1, 0].plot([], [], '-', label="Hanford", c="black")
            indiv_axs[1, 0].plot([], [], '--', label="Livingston", c="black")
            indiv_axs[1, 0].legend(handlelength=3, fontsize=17)
            indiv_axs[1, 0].set_xlabel("Time, (ms)")
            indiv_axs[1, 0].set_ylabel("Final Metric Contribution")

            strain_ts = np.linspace(0, len(strain_chunks[j, 0, :])/SAMPLE_RATE, len(strain_chunks[j, 0, :]))
            indiv_axs[0, 0].plot(strain_ts, strain_chunks[j, 0, :], label = 'Hanford', alpha=0.8)
            indiv_axs[0, 0].plot(strain_ts, strain_chunks[j, 1, :], label = 'Livingston', alpha=0.8)
            indiv_axs[0, 0].set_xlabel('Time, (ms)')
            indiv_axs[0, 0].set_ylabel('strain')
            indiv_axs[0, 0].legend()
            indiv_axs[0, 0].set_title(f'gps time: {start_point + midpoints[loudest]/SAMPLE_RATE + hour_split*eval_at_once_len:.2f}')
            p = midpoints[loudest]

            do_q_scan = True
            if do_q_scan:
                # plot the Q-scans
                left_edge = 1024
                right_edge = 1024
                q_edge = int(7.5*4096)
                H_strain = gwpy_timeseries[0][p-left_edge-q_edge + eval_at_once_len*hour_split*SAMPLE_RATE:p+right_edge+q_edge + eval_at_once_len*hour_split*SAMPLE_RATE]
                L_strain = gwpy_timeseries[1][p-left_edge-q_edge + eval_at_once_len*hour_split*SAMPLE_RATE:p+right_edge+q_edge + eval_at_once_len*hour_split*SAMPLE_RATE]
                t0 = H_strain.t0.value
                dt = H_strain.dt.value

                H_hq = H_strain.q_transform(outseg=(t0+q_edge*dt, t0+q_edge*dt+(left_edge+right_edge)*dt), whiten=False)
                L_hq = L_strain.q_transform(outseg=(t0+q_edge*dt, t0+q_edge*dt+(left_edge+right_edge)*dt), whiten=False)
                f = np.array(H_hq.yindex)
                t = np.array(H_hq.xindex)
                #t=strain_ts *1000
                t -= t[0]

                im_H = indiv_axs[0, 1].pcolormesh(t*1000, f, np.array(H_hq).T, vmax = 25, vmin = 0)
                indiv_fig[0, 1].colorbar(im_H, ax=indiv_axs[0, 1], label = "spectral power")
                indiv_axs[0, 1].set_yscale("log")
                indiv_axs[0, 1].set_xlabel("Time (ms)")
                indiv_axs[0, 1].set_ylabel("Freq (Hz)")
                indiv_axs[0, 1].set_title("Hanford Q-Transform")

                im_L  = indiv_axs[1, 1].pcolormesh(t*1000, f, np.array(L_hq).T, vmax = 25, vmin = 0)
                indiv_fig[1, 1].colorbar(im_L, ax=indiv_axs[1, 1], label = "spectral power")
                indiv_axs[1, 1].set_yscale("log")
                indiv_axs[1, 1].set_xlabel("Time (ms)")
                indiv_axs[1, 1].set_ylabel("Freq (Hz)")
                indiv_axs[1, 1].set_title("Livingston Q-Transform")

            #FINAL_FAR_HISTOGRAM = np.load('/n/home00/emoreno/katya_LITERALLY/gw_anomaly/ryan/FINAL_FINAL_HISTOGRAM.npy')
            #best_far = get_far(filtered_final_score[j], FINAL_FAR_HISTOGRAM)[0]
            best_far = 0
            best_score = fm_scores[j][0] # [ [one element], [one element]] structure
            print("best_score", best_score)
            base = f'{savedir}/{start_point+p/SAMPLE_RATE:.3f}_{best_far}_{best_score:.2f}'
            plt.savefig(base+'.png', dpi=300, bbox_inches="tight")
            print('Saved the plot in', base)
            rename_map = np.array([["strain", "H_qtransform"],["gwak_values", "L_qtransform"]])
            for a in range(2):
                for b in range(2):
                    indiv_fig[a, b].savefig(f"{base}_{rename_map[a, b]}.png", dpi=200)
                    plt.close(indiv_fig[a, b])

            # indexing into midpoints with loudest should carry over to original and recreated - if just taking the strongst point to recreate
            #p = midpoints[loudest]

def plot_strain(start_point, end_point):
    # Load LIGO data
    strainL1 = TimeSeries.get(f'L1:{CHANNEL}', start_point, end_point)
    strainH1 = TimeSeries.get(f'H1:{CHANNEL}', start_point, end_point)
    # Whiten, bandpass, and resample
    sample_rate = 4096
    bandpass_low = 30
    bandpass_high = 1500
    #strainL1 = strainL1.resample(sample_rate)
    strainL1 = strainL1.whiten().bandpass(bandpass_low, bandpass_high)
    strainL1 = strainL1.resample(sample_rate)
    strainH1 = strainH1.whiten().bandpass(bandpass_low, bandpass_high)
    strainH1 = strainH1.resample(sample_rate)
    np.save('strainL1.npy', strainL1[int(100.3*sample_rate):int(100.8*sample_rate)].value)
    np.save('strainH1.npy', strainH1[int(100.3*sample_rate):int(100.8*sample_rate)].value)

    plt.figure()
    plt.plot(strainL1[int(100.3*sample_rate):int(100.8*sample_rate)])
    plt.plot(strainH1[int(100.3*sample_rate):int(100.8*sample_rate)])
    plt.show()

    H_hq = strainH1.q_transform(outseg=(start_point+100, end_point-100), whiten=False)
    L_hq = strainL1.q_transform(outseg=(start_point+100, end_point-100), whiten=False)

    vmax = 25.0
    vmin = 0

    plot = H_hq.plot(figsize=[8, 4], vmax=vmax, vmin=vmin)
    ax = plot.gca()
    ax.set_xscale('seconds')
    ax.set_yscale('log')
    ax.set_ylim(20, 500)
    ax.set_ylabel('Frequency [Hz]')
    ax.grid(True, axis='y', which='both')
    ax.colorbar(cmap='viridis', label='Normalized energy')
    plot.show()

    plot = L_hq.plot(figsize=[8, 4], vmax=vmax, vmin=vmin)
    ax = plot.gca()
    ax.set_xscale('seconds')
    ax.set_yscale('log')
    ax.set_ylim(20, 500)
    ax.set_ylabel('Frequency [Hz]')
    ax.grid(True, axis='y', which='both')
    ax.colorbar(cmap='viridis', label='Normalized energy')
    plot.show()


def main(args):
    try:
        os.makedirs(args.savedir)
    except FileExistsError:
        None

    check_cat1 = False
    check_cat2 = False

    trained_path = "/home/katya.govorkova/gwak-paper-final-models/"
    o3_detection_segments = ['1243305662.9310', '1241104246.7490', '1249635282.3590',
        '1242442957.4230', '1241624696.5500', '1251009253.7240', '1240050919.5040',
        '1250981809.4370', '1238351461.2030', '1246417246.8230', '1246487209.3080',
        '1249035984.2120', '1242827473.3700', '1248280604.5540', '1240878400.3070',
        '1251709954.8220', '1249529264.6980', '1239155734.1820', '1253638396.3240',
        '1251452408.2800', '1260825547.0250', '1265853772.5540', '1260164276.4340',
        '1262676274.8130', '1260258000.1600', '1259852757.6790', '1267617687.9370',
        '1261041997.0680', '1267610457.9930', '1264316116.3950', '1260288309.7230',
        '1263906435.1910', '1263013367.0550', '1267610487.9370', '1262203619.4010',
        '1262002542.6820']

    if check_cat1:
        vetodef = DataQualityDict.from_veto_definer_file('data/H1L1V1-HOFT_C01_V1ONLINE_O3_BURST.xml')
        vetodef.populate()

        h1_cat1 = DataQualityDict({k:v for k, v in vetodef.items() if v.ifo == 'H1' and v.category == 1})
        h1_cat1_flag = h1_cat1.union()

        l1_cat1 = DataQualityDict({k:v for k, v in vetodef.items() if v.ifo == 'L1' and v.category == 1})
        l1_cat1_flag = l1_cat1.union()

        l1_cat1_segments = [(i[0], i[1]) for i in l1_cat1_flag.active]
        h1_cat1_segments = [(i[0], i[1]) for i in h1_cat1_flag.active]

        cat1_segments = []
        for det in o3_detection_segments:
            det = float(det)
            if det<1256655618: det+=10
            if check_segment_overlap(det, l1_cat1_segments) or \
                check_segment_overlap(det, h1_cat1_segments):
                cat1_segments.append(det)
        print('All Cat1 segments are:', cat1_segments)

    if check_cat2:
        l1_cat2_filename = '/home/eric.moreno/QUAK/katya/gw-anomaly/output/L1_CAT2_ACTIVE_SEGMENTS.txt'
        l1_cat2_segments = read_segments_from_file(l1_cat2_filename)

        h1_cat2_filename = '/home/eric.moreno/QUAK/katya/gw-anomaly/output/H1_CAT2_ACTIVE_SEGMENTS.txt'
        h1_cat2_segments = read_segments_from_file(h1_cat2_filename)

        cat2_segments = []
        for det in o3_detection_segments:
            det = float(det)
            if det<1256655618: det+=10
            if check_segment_overlap(det, l1_cat2_segments) or \
                check_segment_overlap(det, h1_cat2_segments):
                cat2_segments.append(det)
        print('All Cat2 segments are:', cat2_segments)
        assert 0

    savedir = args.savedir+'/O3'
    try:
        os.makedirs(savedir)
    except FileExistsError:
        None

    for i, det in enumerate(o3_detection_segments):
        det = float(det)
        if det<1256655618: det+=10

        # first, plot the strain
        # plot_strain(det-100.5,det+100.5)

        A, B = det,det+3600

        H, L, H_rbw, L_rbw = resample_bandpass_whiten(A, B)
        if H is None and L is None: continue
        data = np.vstack([np.array(H.data), np.array(L.data)])
        data_rbw = np.vstack([np.array(H_rbw.data), np.array(L_rbw.data)])

        if data.shape[-1] < 1e5: return None
        get_evals(data, trained_path, savedir, int(A), [H, L],
            data_rbw=data_rbw,
            detection_point=det,
            manual_eval_times=[1000])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--savedir', type=str, default='output/paper')

    args = parser.parse_args()
    main(args)
