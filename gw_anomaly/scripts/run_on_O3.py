import pickle
import time
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
import torch.nn as nn
import pickle
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


def sig_prob_function(evals, scale=40):
    sigmoid = lambda x: 1/(1+np.exp(-x))
    return 1-(sigmoid(scale * (evals-0.5)))


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
        print('Point', point)
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

    strainL1 = strainL1_0.resample(sample_rate).whiten().bandpass(bandpass_low, bandpass_high)
    # strainL1 = strainL1.whiten()
    strainH1 = strainH1_0.resample(sample_rate).whiten().bandpass(bandpass_low, bandpass_high)
    # strainH1 = strainH1.whiten()

    return strainH1, strainL1, strainH1_0, strainL1_0


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


def get_evals(data_, model_path, savedir, start_point, gwpy_timeseries, detection_point=None):

    # split the data into 1-hour chunks to fit in memory best
    eval_at_once_len = int(3600)
    N_one_hour_splits = int(data_.shape[1]//(eval_at_once_len*SAMPLE_RATE) + 1)

    for hour_split in range(N_one_hour_splits):
        start = int(hour_split*SAMPLE_RATE*eval_at_once_len)
        end = int(min(data_.shape[1], (hour_split+1)*SAMPLE_RATE*eval_at_once_len))
        print(start, end)
        if end - 10 < start:
            return None
        data = data_[:, start:end]

        models_path = [f"{MODELS_LOCATION}/bbh.pt",
                       f"{MODELS_LOCATION}/sglf.pt",
                       f"{MODELS_LOCATION}/sghf.pt",
                       f"{MODELS_LOCATION}/background.pt",
                       f"{MODELS_LOCATION}/glitches.pt"]

        gwak_models = load_gwak_models(models_path, DEVICE, GPU_NAME)
        orig_kernel = 50
        kernel_len = int(orig_kernel * 5/SEGMENT_OVERLAP)
        kernel = torch.ones((1, kernel_len)).float().to(DEVICE)/kernel_len
        kernel = kernel[None, :, :]
        heuristics_tests = True

        norm_factors = np.load(f"{FM_LOCATION}/norm_factor_params.npy")

        fm_model_path = (f"{FM_LOCATION}/fm_model.pt")
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
        FAR_2days = -1 # lowest FAR bin we want to worry about

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
                                            DEVICE, window_size=2048) # 0.25 seconds on either side
                                                                    # so it should come out to desired 0.5

        filtered_final_scaled_evals = filtered_final_scaled_evals.detach().cpu().numpy()
        filtered_final_score = filtered_final_score.detach().cpu().numpy()

        print('edge_check_filter', edge_check_filter)

        filtered_final_scaled_evals = filtered_final_scaled_evals[edge_check_filter]
        filtered_final_score = filtered_final_score[edge_check_filter]
        timeslide_chunks = timeslide_chunks[edge_check_filter]
        indices = indices[edge_check_filter]

        filtered_timeslide_chunks = timeslide_chunks


        heuristics_tests = True
        if heuristics_tests:
            # model_path = f"{FM_LOCATION}/model_heuristic.h5"
            model_path = f'/home/eric.moreno/QUAK/ryan_checks/ryan/model.h5'
            model_heuristic = BasedModel().to(DEVICE)
            model_heuristic.load_state_dict(torch.load(model_path, map_location=DEVICE))

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
                print('final_final[0]',final_final[0])
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

        l1_cat2_filename = '/home/eric.moreno/QUAK/katya/gw-anomaly/output/L1_CAT2_ACTIVE_SEGMENTS.txt'
        l1_cat2_segments = read_segments_from_file(l1_cat2_filename)

        h1_cat2_filename = '/home/eric.moreno/QUAK/katya/gw-anomaly/output/H1_CAT2_ACTIVE_SEGMENTS.txt'
        h1_cat2_segments = read_segments_from_file(h1_cat2_filename)


        cat2_name = '_cat2' if check_segment_overlap(detection_point, l1_cat2_segments) or check_segment_overlap(detection_point, h1_cat2_segments) else ''

        scaled_evals = scaled_evals.cpu().numpy()
        scaled_evals = combine_freqcorr(scaled_evals)
        bias_value = bias_value.cpu().numpy()
        smoothed_scores = smoothed_scores.cpu().numpy()
        for j in range(len(gwak_values)):
            fig, axs = plt.subplots(2, 2, figsize=(12, 11))
            loudest = indices[j]
            left_edge = 1024 // SEGMENT_OVERLAP
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
            # added 10 sec to aligned the GPS time
            axs[0, 0].set_title(f'GPS time: {10+start_point + midpoints[loudest]/SAMPLE_RATE + hour_split*eval_at_once_len:.3f}')
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

            best_score = fm_scores[j][0] # [ [one element], [one element]] structure
            plt.savefig(f'{savedir}/{start_point+p/SAMPLE_RATE:.3f}_{best_score:.2f}{cat2_name}.png', bbox_inches="tight")
            pickle.dump(axs, open(f'{savedir}/{start_point+p/SAMPLE_RATE:.3f}_{best_score:.2f}{cat2_name}.pickle', 'wb'))
            print(f'Found detection {savedir}/{start_point+p/SAMPLE_RATE:.3f}_{best_score:.2f}{cat2_name}')
            plt.close()


def main(args):
    try:
        os.makedirs(args.savedir)
    except FileExistsError:
        None

    trained_path = "/home/katya.govorkova/gw_anomaly/output/O3av2/"

    run_short_test = False
    if run_short_test:
        start = time.time()
        # testing code
        A = 1243303084
        B = A + 3600

        Hclean, Lclean, Hraw, Lraw = resample_bandpass_whiten(A, B)
        data = np.vstack([np.array(Hclean.data), np.array(Lclean.data)])

        get_evals(data, trained_path, args.savedir, int(A), [Hclean, Lclean], detection_point=float(A))
        end = time.time()
        print('Time to evaluate one hour of zero lag', end - start, 'sec')

        assert 0

    valid_segments = ['1243305662.9310', '1241104246.7490', '1249635282.3590',
        '1242442957.4230', '1241624696.5500', '1251009253.7240', '1240050919.5040',
        '1250981809.4370', '1238351461.2030', '1246417246.8230', '1246487209.3080',
        '1249035984.2120', '1242827473.3700', '1248280604.5540', '1240878400.3070',
        '1251709954.8220', '1249529264.6980', '1239155734.1820', '1253638396.3240',
        '1251452408.2800', '1260825547.0250', '1265853772.5540', '1260164276.4340',
        '1262676274.8130', '1260258000.1600', '1259852757.6790', '1267617687.9370',
        '1261041997.0680', '1267610457.9930', '1264316116.3950', '1260288309.7230',
        '1263906435.1910', '1263013367.0550', '1267610487.9370', '1262203619.4010',
        '1262002542.6820']
    savedir = args.savedir+'O3'
    try:
        os.makedirs(savedir)
    except FileExistsError:
        None
    for det in valid_segments:

        det = float(det)
        if det<1256655618: det+=10

        print(f'Analyzing {det}')

        A, B = float(det)-3600/2,float(det)+3600/2

        H, L, _, _ = resample_bandpass_whiten(A, B)
        if H is None and L is None: continue
        data = np.vstack([np.array(H.data), np.array(L.data)])

        if data.shape[-1] < 1e5: return None
        get_evals(data, trained_path, savedir, int(A), [H, L], detection_point=float(det))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--savedir', type=str, default='output/paper')

    args = parser.parse_args()
    main(args)
