import os
import numpy as np
import argparse
import torch
from gwpy.timeseries import TimeSeries
from astropy import units as u
import matplotlib.pyplot as plt
from models import LinearModel, GwakClassifier
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
    return 1-(sigmoid(scale * (evals-0.5)))
        
def get_far(score, sort_eval):
    ind = np.searchsorted(sort_eval, score)
    if ind == len(sort_eval):
        ind -= 1
    #N = len(sort_eval)
    units = 10000*3.15e7
    return ind/units

# def make_metric_vs_far(savepath, bins_path=f"/home/katya.govorkova/gwak-paper-final-models/far_bins_50.npy"):
#     #far_bar = far_to_metric(3600*24*2, far_bins)
#     search_vals = np.logspace(-3, np.log10(100), 200) # in number per month
#     search_vals *= 30*24*3600

#     metric_vals = []
#     far_bins = np.load(bins_path)
#     for elem in search_vals:
#         metric_vals.append(far_to_metric(elem, far_bins))



#     #plt.plot(search_vals/(30*24*3600), metric_vals)
#     plt.xscale("log")
#     plt.plot(1/search_vals, metric_vals)

#     vbars = [3600, 24*3600, 7*24*3600, 30*24*3600]
#     labels = ["hour", "day", "week", "month"]
#     for i, val in enumerate(vbars):
#         plt.axvline(x=1/val, label=labels[i], alpha=0.9**(len(labels)-i), c="black")
#     plt.legend()
#     plt.xlabel("FAR, Hz")
#     plt.ylabel("corresponding metric values")
#     plt.savefig(f"{savepath}/FAR_vs_metric.pdf", dpi=300)
#     return 1/search_vals, metric_vals

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

def whiten_bandpass_resample(
        start_point,
        end_point,
        sample_rate=SAMPLE_RATE,
        bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH,
        shift=None):

    device = torch.device(GPU_NAME)

    # Load LIGO data
    #try:
    # Load LIGO data
    try:
        start_point, end_point = int(start_point)+10, int(end_point)-10
        strainL1 = TimeSeries.get(f'L1:{CHANNEL}', start_point, end_point)
        strainH1 = TimeSeries.get(f'H1:{CHANNEL}', start_point, end_point)
        #else:
        #    strainL1 = TimeSeries.fetch(f'L1:{CHANNEL}', start_point, end_point)
        #    strainH1 = TimeSeries.fetch(f'H1:{CHANNEL}', start_point, end_point)
            
        #strainL1 = TimeSeries.get(f'L1:{CHANNEL}', start_point, end_point, host="losc-nds.ligo.org")
        #strainH1 = TimeSeries.get(f'H1:{CHANNEL}', start_point, end_point, host="losc-nds.ligo.org") #.get, verbose,,, .find
            # Save with pickle
            #open the pickle files    
            #with open(f"/n/holyscratch01/iaifi_lab/emoreno/gwak_o3a/data/L1_{start_point}_{end_point}.pkl", 'rb') as f:
            #    strainL1 = pickle.load(f)
            #
            #with open(f"/n/holyscratch01/iaifi_lab/emoreno/gwak_o3a/data/H1_{start_point}_{end_point}.pkl", 'rb') as f:
            #    strainH1 = pickle.load(f)

        #except:
        #    print(f'Couldnt load {start_point}, {end_point}')
        #    print('SKIPPING')
        #    return None, None

        t0 = int(strainL1.t0 / u.s)

    # if shift != None:
    #     shift_datapoints = shift * sample_rate
    #     temp = strainL1[-shift_datapoints:]
    #     strainL1[shift_datapoints:] = strainL1[:-shift_datapoints]
    #     strainL1[:shift_datapoints] = temp

        # Whiten, bandpass, and resample
        strainL1 = strainL1.resample(sample_rate).bandpass(bandpass_low, bandpass_high)
        strainL1 = strainL1.whiten()

        strainH1 = strainH1.resample(sample_rate).bandpass(bandpass_low, bandpass_high)
        strainH1 = strainH1.whiten()

        return [strainH1, strainL1]
    except:
        return None, None

def whiten_bandpass_resample_new_order(
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


    t0 = int(strainL1.t0 / u.s)

    strainL1 = strainL1_0.bandpass(bandpass_low, bandpass_high).resample(sample_rate).whiten()
    strainH1 = strainH1_0.bandpass(bandpass_low, bandpass_high).resample(sample_rate).whiten()

    return strainH1, strainL1, strainH1_0, strainL1_0


def get_evals(data_, model_path, savedir, start_point, gwpy_timeseries, neworder_clean=None, neworder_raw=None):
    #model_path = '/n/home00/emoreno/katya_LITERALLY/gw_anomaly/ryan/model.h5'
    model_path = "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/plots/model.h5"
    model_heuristic = BasedModel().to(DEVICE)
    model_heuristic.load_state_dict(torch.load(model_path, map_location=DEVICE))


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

        #DEVICE = torch.device(f'cuda:{args.gpu}')
        #model_path = ["/n/home00/emoreno/gw-anomaly/output/gwak-paper-final-models/trained/models/bbh.pt",
        #            "/n/home00/emoreno/gw-anomaly/output/gwak-paper-final-models/trained/models/sglf.pt",
        #            "/n/home00/emoreno/gw-anomaly/output/gwak-paper-final-models/trained/models/sghf.pt",
        #            "/n/home00/emoreno/gw-anomaly/output/gwak-paper-final-models/trained/models/background.pt",
        #                "/n/home00/emoreno/gw-anomaly/output/gwak-paper-final-models/trained/models/glitches.pt"]
        model_path = ["output/O3av2/trained/models/bbh.pt", 
                    "output/O3av2/trained/models/sglf.pt", 
                    "output/O3av2/trained/models/sghf.pt", 
                    "output/O3av2/trained/models/background.pt",
                        "output/O3av2/trained/models/glitches.pt"]

        MODELS_LOCATION = "/home/katya.govorkova/gwak-paper-final-models/trained/models/"
        models_path = [os.path.join(MODELS_LOCATION, os.path.basename(f)) for f in model_path]
        gwak_models = load_gwak_models(models_path, DEVICE, GPU_NAME)
        orig_kernel = 50
        kernel_len = int(orig_kernel * 5/SEGMENT_OVERLAP)
        kernel = torch.ones((1, kernel_len)).float().to(DEVICE)/kernel_len
        kernel = kernel[None, :, :]
        heuristics_tests = True
        #if heuristics_tests:
            #long_relation = np.load("/home/ryan.raikman/share/gwak/long_relation.npy")
            #short_relation = np.load("/home/ryan.raikman/share/gwak/short_relation.npy")
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

        #final_values, midpoints = full_evaluation(
        #                data[None, :, :], models_path, DEVICE,
        #                return_midpoints=True, loaded_models=gwak_models, grad_flag=False)
        final_values, midpoints, original, recreated = full_evaluation(  
                        data[None, :, :], models_path, DEVICE,
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
                #passed_heuristics.append(res<0.46)

                res_sigmoid = sig_prob_function(res)
                print(res, res_sigmoid, filtered_final_score[i])
                final_final = res_sigmoid * filtered_final_score[i]
                print(res, res_sigmoid, filtered_final_score[i])
                filtered_final_score[i] *= res_sigmoid
                print(res, res_sigmoid, filtered_final_score[i])
                passed_heuristics.append(final_final[0] < -1) #[0] since it was saving arrays(arrays)
                

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
            fig, axs = plt.subplots(2, 2, figsize=(28, 17))
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

                    axs[1, 0].plot(1000*quak_evals_ts, scaled_evals[loudest-left_edge:loudest+right_edge, i],
                                label = labels[i], c=cols[i//2], linestyle=line_type)
                else:
                    axs[1, 0].plot(1000*quak_evals_ts, scaled_evals[loudest-left_edge:loudest+right_edge, i],
                                    c=cols[i//2], linestyle=line_type)


            axs[1, 0].plot(1000*quak_evals_ts, smoothed_scores[loudest-left_edge:loudest+right_edge]-bias_value, label = 'final metric', c='black')
            axs[1, 0].plot([], [], '-', label="Hanford", c="black")
            axs[1, 0].plot([], [], '--', label="Livingston", c="black")
            axs[1, 0].legend(handlelength=3, fontsize=17)
            axs[1, 0].set_xlabel("Time, (ms)")
            axs[1, 0].set_ylabel("Final Metric Contribution")

            strain_ts = np.linspace(0, len(strain_chunks[j, 0, :])/SAMPLE_RATE, len(strain_chunks[j, 0, :]))
            axs[0, 0].plot(strain_ts, strain_chunks[j, 0, :], label = 'Hanford', alpha=0.8)
            axs[0, 0].plot(strain_ts, strain_chunks[j, 1, :], label = 'Livingston', alpha=0.8)
            axs[0, 0].set_xlabel('Time, (ms)')
            axs[0, 0].set_ylabel('strain')
            axs[0, 0].legend()
            axs[0, 0].set_title(f'gps time: {start_point} + {midpoints[loudest]/SAMPLE_RATE + hour_split*eval_at_once_len:.3f}')
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

                im_H = axs[0, 1].pcolormesh(t*1000, f, np.array(H_hq).T, vmax = 25, vmin = 0)
                fig.colorbar(im_H, ax=axs[0, 1], label = "spectral power")
                axs[0, 1].set_yscale("log")
                axs[0, 1].set_xlabel("Time (ms)")
                axs[0, 1].set_ylabel("Freq (Hz)")
                axs[0, 1].set_title("Hanford Q-Transform")

                im_L  = axs[1, 1].pcolormesh(t*1000, f, np.array(L_hq).T, vmax = 25, vmin = 0)
                fig.colorbar(im_L, ax=axs[1, 1], label = "spectral power")
                axs[1, 1].set_yscale("log")
                axs[1, 1].set_xlabel("Time (ms)")
                axs[1, 1].set_ylabel("Freq (Hz)")
                axs[1, 1].set_title("Livingston Q-Transform")

            #FINAL_FAR_HISTOGRAM = np.load('/n/home00/emoreno/katya_LITERALLY/gw_anomaly/ryan/FINAL_FINAL_HISTOGRAM.npy')
            #best_far = get_far(filtered_final_score[j], FINAL_FAR_HISTOGRAM)[0]
            best_far = 0
            best_score = fm_scores[j][0] # [ [one element], [one element]] structure
            print("best_score", best_score)
            plt.savefig(f'{savedir}/{start_point+p/SAMPLE_RATE:.3f}_{best_far}_{best_score:.2f}.png', dpi=300, bbox_inches="tight")
            plt.close()

            # indexing into midpoints with loudest should carry over to original and recreated - if just taking the strongst point to recreate
            #p = midpoints[loudest]
            if 0:
                fig, axs = plt.subplots(5, 2, figsize=(10, 13))

                print(494, original['bbh'].shape)
                for n, class_name in enumerate(CLASS_ORDER):
                    
                    #i, j = n // 2, n % 2 
                    i=n
                    detecs = {0:"_hanford", 1:"_livingston"}
                    for j in range(2):
                        axs[i, j].plot(original[class_name][loudest][j], label = f'input_{class_name}')
                        axs[i, j].plot(recreated[class_name][loudest][j], label = f'recreated_{class_name}')
                        #axs[i, j].set_title(class_name + detecs[j])
                        axs[i, j].legend()

                axs[0, 0].set_title("hanford")
                axs[0, 1].set_title("livingston")
                #print("best_score", best_score)
                plt.savefig(f'{savedir}/{start_point+p/SAMPLE_RATE:.3f}_{best_far}_{best_score:.2f}_RECREATIONS.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print(514, scaled_evals[loudest])
                fig, axs = plt.subplots(10, 2, figsize=(10, 26))
                for w in range(-5, 5):
                    best_channel = "bbh"
                    offset = 5
                    axs[w+offset, 0].plot(original[best_channel][loudest+w][0], label = "original")
                    axs[w+offset, 0].plot(recreated[best_channel][loudest+w][0], label = "recreated")
                    axs[w+offset, 0].legend()

                    axs[w+offset, 1].plot(original[best_channel][loudest+w][1], label = "original")
                    axs[w+offset, 1].plot(recreated[best_channel][loudest+w][1], label = "recreated")
                    axs[w+offset, 1].legend()

                    axs[0, 0].set_title("Hanford")
                    axs[0, 1].set_title("Livingston")

                plt.savefig(f'{savedir}/{start_point+p/SAMPLE_RATE:.3f}_{best_far}_{best_score:.2f}_PRODECURAL_RECREATIONS.png', dpi=300, bbox_inches='tight')
                plt.close()


def main(args):
    try:
        os.makedirs(args.savedir)
    except FileExistsError:
        None
    #gwtc_events=parse_gwtc_catalog("/n/home00/emoreno/katya_LITERALLY/gw_anomaly/ryan/gwtc.csv",
    #                              1238166018, 1253977218)

    #valid_segments = np.load("/n/home00/emoreno/gw-anomaly/output/gwak-paper-final-models/O3a_intersections.npy")
    valid_segments = np.load("/home/katya.govorkova/gw-anomaly/output/O3a_intersections.npy")
    #trained_path = "/n/home00/emoreno/gw-anomaly/output/gwak-paper-final-models/" # fix hardcoding later
    trained_path = "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/gwak-paper-final-models/"

    do_only_anomaly = True
    if do_only_anomaly:
        anomaly_start_times = [1243303084, 1240875861]
        for A in anomaly_start_times:
            B = A + 3600
            H, L = whiten_bandpass_resample(A, B)
            
            Hclean, Lclean, Hraw, Lraw = whiten_bandpass_resample_new_order(A, B)
            data = np.vstack([np.array(H.data), np.array(L.data)])

            get_evals(data, trained_path, args.savedir, int(A), [Hclean, Lclean])

        return None

    run_short_test = False
    if run_short_test:
        # testing code
        #print("valid segments", valid_segments)
        #A = 1243303084
        #A = 1240700690
        #A = 1244936268
        A = 1245155831
        A = 1242437036 + 5000
        A = 1246261450
        A = 1246453144+33000
        A = 1240872261+5000
        #A = 1240700690
        print(np.where(valid_segments==A))
        A, B = (valid_segments[143])
        #assert 0
        #print(562, A, B)
       # B-= 1000
        #A += 1000
        #B = A + 10000
       # B = A + 3600

        #A = 1243305672.9
        #A = 1251008449
        #A = 1242440636
        #A = 1240316625
        #B = A + 3600#*3//2
        A = 1240700690.0
        B = 1240735662.0
        A = B-3600

        H, L = whiten_bandpass_resample(A, B)
        Hclean, Lclean, Hraw, Lraw = whiten_bandpass_resample_new_order(A, B)
        data = np.vstack([np.array(H.data), np.array(L.data)])

        get_evals(data, trained_path, args.savedir, int(A), [Hclean, Lclean], clean = )
        assert 0

    f_segments = 'output/done_O3a_segments.npy'
    if not os.path.exists(f_segments):
        np.save(f_segments, np.array([0]))

    for a, b in valid_segments:
        done_segments = np.load(f_segments)
        As = np.arange(a, b, 3600)
        for A in As:
            B = A + 3600
            if B > b: continue
            if A in done_segments and B in done_segments: 
                print("already finished,", A, B)
                continue
            #if B - A >= 1800:
            #A = 1243303084 - 3600
            #B = A + 3600 + 3600 + 3600
            print("Working on", A, B)
            H, L = whiten_bandpass_resample(A, B)
            if H is None and L is None: continue
            data = np.vstack([np.array(H.data), np.array(L.data)])
            print(data.shape)
            if data.shape[-1] < 1e5: return None
            get_evals(data, trained_path, args.savedir, int(A), [H, L])
            np.save(f_segments, np.concatenate((done_segments, [A], [B])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('savedir', type=str, default=None,
                        help='File with valid segments')

    args = parser.parse_args()
    main(args)
