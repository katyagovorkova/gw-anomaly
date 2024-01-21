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
from config import (
    CHANNEL,
    GPU_NAME,
    SEGMENT_OVERLAP,
    SAMPLE_RATE,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    FACTORS_NOT_USED_FOR_FM,
    MODELS_LOCATION,
    SEG_NUM_TIMESTEPS
    )
from helper_functions import far_to_metric, compute_fars, load_gwak_models
DEVICE = torch.device(GPU_NAME)
from scipy.signal import welch
heuristics_tests = True
if heuristics_tests: #define the heuristic test helper functions
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

    def compute_signal_strength(x):
        psd = welch(x, axis=1)[1]
        HLS = np.log(np.sum(psd[0]))
        LLS = np.log(np.sum(psd[1]))
        return (HLS+LLS)/2

    def compute_signal_strength_chop(x, y):
        psd0 = welch(x)[1]
        psd1 = welch(y)[1]
        HLS = np.log(np.sum(psd0))
        LLS = np.log(np.sum(psd1))
        return (HLS+LLS)/2

    def make_split_idxs(duration, seglen=200, overlap=50):
        top = seglen
        splits = []
        while top < duration:
            splits.append([top-seglen, top])
            top += overlap
        return splits

    def parse_strain(x):
        # take strain, compute the long sig strenght & pearson
        # split it up, do the same thing for short
        long_pearson, shift_idx = shifted_pearson(x[0], x[1], 50, len(x[0])-50)
        long_sig_strength = compute_signal_strength_chop(x[0, 50:-50], x[1, 50+shift_idx:len(x[0])-50+shift_idx] )


        split_idxs = make_split_idxs(len(x[0]))[1:-1] # cut out ends to shifting can happen there
        short_pearsons = []
        short_sig_strengths = []
        for split in split_idxs:
            start, end = split

            pearson, shift_idx = shifted_pearson(x[0], x[1], start, end)
            sig_strength = compute_signal_strength_chop(x[0, start:end], x[1, start+shift_idx:end+shift_idx] )

            short_pearsons.append(pearson)
            short_sig_strengths.append(sig_strength)

        short_pearsons = np.array(short_pearsons)
        short_sig_strengths = np.array(short_sig_strengths)

        return [long_sig_strength, long_pearson], [short_sig_strengths, short_pearsons]

    def combine_freqcorr(x):
        # x shape is (N, 16)
        new = np.zeros((x.shape[0], 11))
        jump = 0
        for i in range(15):
            if i % 3 != 2:
                new[:, i-jump] = x[:, i]
            else:
                jump += 1
                new[:, -1] += x[:, i]

        #new[:, -1] = x[:, -1]

        return new

    def compute_required_pearson(strens, relation):
        # relation is [:, 0] strengths, [:, 1] pearsons
        required_pearsons = []
        flag = False
        if not isinstance(strens, np.ndarray) and not type(strens) is list :
            strens = [strens]
            flag = True
        for stren in strens:
            idx = np.searchsorted(relation[:, 0], stren)
            if idx == len(relation[:, 0]):
                required_pearsons.append(-1)
            else:
                required_pearsons.append(relation[idx, 1])
        if flag:
            return required_pearsons[0]
        return required_pearsons

    def single_condition(required, actual, bar=0.15, debug=False):
        # does it pass?
        if debug:
            print(f"required: {required}, actual: {actual}")
        return  required + bar > actual

    def iterated_condition(required, actual, failed_fraction=0.2, debug=False):
        # does it pass?
        failed_count = 0
        for i in range(len(required)):
            if not single_condition(required[i], actual[i], debug=debug):
                failed_count += 1

        return failed_count / len(required) < failed_fraction

    def pairwise_symmetry_condition(scores):
        # does not include glitch feature
        def functional(x, y):
            #enforce  |x| >= |y|
            x = abs(x)
            y = abs(y)

            if y > x:
                x, y = y, x
            # signal autoencoder feature shouldn't be significantly positive
            return (x+1.5)/(y+1.5) - 1.5
        passed = True
        for pair in [[2, 3], [6, 7], [8, 9]]:
            a, b = pair

            if functional(scores[a], scores[b]) > 0:
                passed = False

        return passed

    def joint_heuristic_test(strain, gwak_features, short_relation, long_relation):
        long, short = parse_strain(strain)
        # for the short range
        short_required_pearson = compute_required_pearson(short[0], short_relation)

        # for the long range
        long_required_pearson = compute_required_pearson(long[0], long_relation)
        return pairwise_symmetry_condition(gwak_features) and single_condition(long_required_pearson, long[1]) and iterated_condition(short_required_pearson, short[1])

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
        
def make_metric_vs_far(savepath, bins_path=f"/home/katya.govorkova/gwak-paper-final-models/far_bins_50.npy"):
    #far_bar = far_to_metric(3600*24*2, far_bins)
    search_vals = np.logspace(-3, np.log10(100), 200) # in number per month
    search_vals *= 30*24*3600

    metric_vals = []
    far_bins = np.load(bins_path)
    for elem in search_vals:
        metric_vals.append(far_to_metric(elem, far_bins))



    #plt.plot(search_vals/(30*24*3600), metric_vals)
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
        savedir,
        sample_rate=SAMPLE_RATE,
        bandpass_low=BANDPASS_LOW,
        bandpass_high=BANDPASS_HIGH,
        shift=None):

    device = torch.device(GPU_NAME)

    # Load LIGO data
    strainL1 = TimeSeries.get(f'L1:{CHANNEL}', start_point, end_point)
    strainH1 = TimeSeries.get(f'H1:{CHANNEL}', start_point, end_point) #.get, verbose,,, .find

    t0 = int(strainL1.t0 / u.s)
    print(t0)

    if shift != None:
        shift_datapoints = shift * sample_rate
        temp = strainL1[-shift_datapoints:]
        strainL1[shift_datapoints:] = strainL1[:-shift_datapoints]
        strainL1[:shift_datapoints] = temp

    # Whiten, bandpass, and resample
    strainL1 = strainL1.resample(sample_rate)
    strainL1 = strainL1.whiten().bandpass(bandpass_low, bandpass_high)
    
    strainH1 = strainH1.resample(sample_rate)
    strainH1 = strainH1.whiten().bandpass(bandpass_low, bandpass_high)
        
    return [strainH1, strainL1]

def get_evals(data_, model_path, savedir, start_point, gwpy_timeseries):
    # split the data into 1-hour chunks to fit in memory best
    print(data_.shape)
    N_one_hour_splits = data_.shape[1]//(3600*SAMPLE_RATE) + 1

    for hour_split in range(N_one_hour_splits):
        start = int(hour_split*SAMPLE_RATE*3600)
        end = int(min(data_.shape[1], (hour_split+1)*SAMPLE_RATE*3600))

        data = data_[:, start:end]
        
        #DEVICE = torch.device(f'cuda:{args.gpu}')
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
        if heuristics_tests:
            long_relation = np.load("/home/ryan.raikman/share/gwak/long_relation.npy")
            short_relation = np.load("/home/ryan.raikman/share/gwak/short_relation.npy")
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

        final_values, midpoints = full_evaluation(
                        data[None, :, :], models_path, DEVICE, 
                        return_midpoints=True, loaded_models=gwak_models, grad_flag=False)
        
        final_values = final_values[0]
        FAR_2days = -2 # lowest FAR bin we want to worry about

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
        
        if len(indices) == 0: return None  
        

        indices = event_clustering(indices, smoothed_scores, 5*SAMPLE_RATE/SEGMENT_OVERLAP, DEVICE) # 5 seconds
        filtered_final_score = smoothed_scores.index_select(0, indices)
        filtered_final_scaled_evals = scaled_evals.index_select(0, indices)

        

        indices = indices.detach().cpu().numpy()
        # extract important "events" with indices
        timeslide_chunks, edge_check_filter = extract_chunks(data, 0, 
                                            midpoints[indices],
                                            DEVICE, window_size=1024) # 0.25 seconds on either side
                                                                    # so it should come out to desired 0.5
        filtered_final_scaled_evals = filtered_final_scaled_evals.detach().cpu().numpy()
        filtered_final_score = filtered_final_score.detach().cpu().numpy()

        filtered_final_scaled_evals = filtered_final_scaled_evals[edge_check_filter]
        filtered_final_score = filtered_final_score[edge_check_filter]
        timeslide_chunks = timeslide_chunks[edge_check_filter]
        indices = indices[edge_check_filter]

        #print("386", filtered_final_scaled_evals.shape, timeslide_chunks.shape)
        filtered_timeslide_chunks = timeslide_chunks
        if heuristics_tests:
            N_initial = len(filtered_final_score)
            combined_freqcorr = combine_freqcorr(filtered_final_scaled_evals)
            passed_heuristics = []
            for i, strain_segment in enumerate(timeslide_chunks):
                passed_heuristics.append(joint_heuristic_test(strain_segment, combined_freqcorr[i],
                                                            short_relation, long_relation))
                
            #print("passed heuristics:", passed_heuristics)

            filtered_final_scaled_evals = filtered_final_scaled_evals[passed_heuristics]
            filtered_final_score = filtered_final_score[passed_heuristics]
            filtered_timeslide_chunks = timeslide_chunks[passed_heuristics]
            indices = indices[passed_heuristics]

            print(f"Fraction removed by heuristics test {N_initial -len(filtered_final_score)}/{N_initial}")
        # rename them for less confusion, easier typing
        gwak_values = filtered_final_scaled_evals
        fm_scores = filtered_final_score
        strain_chunks = filtered_timeslide_chunks

        if strain_chunks.shape[0] == 0: return None
        # plotting all these significant events
        n_points = strain_chunks.shape[2]

        quak_evals_ts = np.linspace(0, 2*(1024//SAMPLE_RATE)*SEGMENT_OVERLAP/SAMPLE_RATE , 2*1024//SAMPLE_RATE)
        scaled_evals = scaled_evals.cpu().numpy()
        scaled_evals = combine_freqcorr(scaled_evals)
        bias_value = bias_value.cpu().numpy()
        smoothed_scores = smoothed_scores.cpu().numpy()
        for j in range(len(gwak_values)):
            fig, axs = plt.subplots(2, 2, figsize=(28, 17))
            loudest = indices[j]
            left_edge = 1024 //SEGMENT_OVERLAP
            right_edge = 1024 // SEGMENT_OVERLAP
            labels = ['background','background', 'bbh','bbh', 'glitch', 'glitch', 'sglf', 'sglf', 'sghf', 'sghf', 'freq corr']
            cols = ['purple', 'blue', 'green', 'salmon', 'goldenrod', 'brown' ]
            for i in range(scaled_evals.shape[1]):
                #print(scaled_evals_.shape)
                #print(loudest-left_edge, loudest+right_edge, i)
                line_type = "-"
                if i% 2 == 1:
                    line_type = "--"    
                if i % 2 == 0 or labels[i] in ["freq corr"]:

                    axs[1, 0].plot(scaled_evals[loudest-left_edge:loudest+right_edge, i], 
                                label = labels[i], c=cols[i//2], linestyle=line_type)
                else:
                    axs[1, 0].plot(scaled_evals[loudest-left_edge:loudest+right_edge, i], 
                                    c=cols[i//2], linestyle=line_type)

            
            axs[1, 0].plot( smoothed_scores[loudest-left_edge:loudest+right_edge]-bias_value, label = 'final metric', c='black')
            axs[1, 0].plot([], [], '-', label="Hanford", c="black")
            axs[1, 0].plot([], [], '--', label="Livingston", c="black")
            axs[1, 0].legend(handlelength=3, fontsize=17)
            axs[1, 0].set_xlabel("Time, (ms)")
            axs[1, 0].set_ylabel("Final Metric Contribution")

            strain_ts = np.linspace(0, (right_edge+left_edge)/SAMPLE_RATE, right_edge+left_edge)
            #print("190", strain_orig.shape, p-left_edge,p+right_edge)
            axs[0, 0].plot(strain_chunks[j, 0, :], label = 'Hanford', alpha=0.8)
            axs[0, 0].plot(strain_chunks[j, 1, :], label = 'Livingston', alpha=0.8)
            axs[0, 0].set_xlabel('Time, (ms)')
            axs[0, 0].set_ylabel('strain')
            axs[0, 0].legend()
            axs[0, 0].set_title(f'gps time: {start_point} + {midpoints[loudest]/SAMPLE_RATE + hour_split*3600:.3f}')
            #gps_times.append(p/SAMPLE_RATE + start_point)
            p = midpoints[loudest]
            if 0:
                # plot the Q-scans
                
                q_edge = int(7.5*4096)
                H_strain = gwpy_timeseries[0][p-left_edge-q_edge:p+right_edge+q_edge + 3600*hour_split*SAMPLE_RATE]
                L_strain = gwpy_timeseries[1][p-left_edge-q_edge:p+right_edge+q_edge + 3600*hour_split*SAMPLE_RATE]
                t0 = H_strain.t0.value
                dt = H_strain.dt.value
                #print(p-left_edge-q_edge,p+right_edge+q_edge)
                #print(p, left_edge, q_edge, t0+q_edge*dt, t0+q_edge*dt+(left_edge+right_edge)*dt)
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
                
                #{savedir}/{start_point}/
            best_far = "NOFAR"
            best_score = fm_scores[j][0]
            print("best_score", best_score)
            extra = "NOEXTRA"
            plt.savefig(f'./{start_point+p/SAMPLE_RATE:.3f}_{best_far}_{best_score:.2f}{extra}.png', dpi=300, bbox_inches="tight")
            plt.close()

def main():
    gwtc_events=parse_gwtc_catalog("/home/ryan.raikman/s22/forks/katya/gw-anomaly/data/gwtc.csv", 
                                  1238166018, 1253977218)

    valid_segments = np.load("/home/katya.govorkova/gwak-paper-final-models/O3a_intersections.npy")
    trained_path = "/home/katya.govorkova/gwak-paper-final-models/" # fix hardcoding later
    #trained_path = "/home/katya.govorkova/gw-anomaly/output/O3av2_non_linear_bbh_only/"
    savedir = 'output/O3a_runthrough/'
    A = 1243303084
    B = A + 3600*3//2
    #if 0:
    #H, L = whiten_bandpass_resample(A, B, savedir)
    #data = np.vstack([np.array(H.data), np.array(L.data)])

    #np.save("temporary_save.npy", data)
    data = np.load("temporary_save.npy")
    H, L = None, None
    #def get_evals(data_, model_path, from_saved_models, savedir, start_point):
    get_evals(data, trained_path, savedir, int(A), [H, L])
    assert 0



    if 0:
        A = 1243303084
        B = A + 3600
        data = whiten_bandpass_resample(A, B, savedir)
        get_evals(data, trained_path, savedir, int(A))
             


    if 0:
        A, B = None,  None
        target = 1249852257.0
        target = 1245955943.1
        target = 1249852257.0	
        target = 1242459857.4	
        low, high = valid_segments[0][0], valid_segments[-1][1]
        #print(target-low, high-target)
        for seg in valid_segments:
            a, b = seg
            if a < target and b>target:
                A = str(a); B = str(b)
                break
        #print(A, B)
        B = target + 100
        A = target - 1050
        print("A, B", A, B)
        #reduce
        #assert 0
        
        data = whiten_bandpass_resample(A, B, savedir)
        get_evals(data, trained_path, savedir, int(A))
    #/home/ryan.raikman/s22/forks/katya/gw-anomaly/scripts/output/all_O3a/
    if 0:
        for seg in valid_segments:
            
            a, b = seg
            A, B = str(a), str(b)
            #if int(A) == 1238677588:
            #if int(A) == 1238645904:
            #if int(A) == 1247614196:
            #if int(A) == 1239080462:
            #if int(A) == 1242441167:
            #if int(A) == 1242458057:
            if b-a > 500:
                chunks = make_eval_chunks(a, b, 3600)
                for A, B in chunks:
                    data = whiten_bandpass_resample(A, B, savedir)
                    get_evals(data, trained_path, savedir, int(A))
                
    if 0:
        base = 1242459857
        a, b = base-1800, base+1800
        A, B = str(a), str(b)
        #chunks = make_eval_chunks(a, b, 3600)
        #for A, B in chunks:
        data = whiten_bandpass_resample(A, B, savedir)
        get_evals(data, trained_path, savedir, int(A), far_search=[far_a, far_b])

    if 0:
        segments_to_analyze = []
        SNRs = []
        gw_event_times = []
        for i, gps_time in enumerate(gwtc_events[:, 0]):
            out = find_segment(gps_time, valid_segments)
            if out is not None:
                a, b = out
                if b-a > 3600:
                    # want exactly 1 hour of data for consistent whitening
                    low = max(gps_time-1800, a)
                    upper = 0
                    if low == a:
                        upper += 1800-(gps_time-a)
                    high = min(gps_time+1800+upper, b)

                    segments_to_analyze.append([low, high])
                    SNRs.append(gwtc_events[i, 1])
                    gw_event_times.append(gps_time)

        for i, (a, b) in enumerate(segments_to_analyze):
            a, b = int(a), int(b)
            A, B = str(a), str(b)
            # split up into one-hour chunks, to standardize the whitenening window
            
            data = whiten_bandpass_resample(A, B, savedir, shift=None)
            get_evals(data, trained_path, savedir, int(A), extra=f"{gw_event_times[i]}_SNR{SNRs[i]}")

    if 0:
        _STRAIN_START = 1238166018 # for O3b 1256663958 1238166018
        _STRAIN_STOP = 1238170289 # for O3b 1256673192 1238170289
        _STRAIN_START = 1242442967-1800
        _STRAIN_STOP = 1242442967 + 1800
        a, b = _STRAIN_START, _STRAIN_STOP
        A, B = str(a), str(b)
        data = whiten_bandpass_resample(A, B, savedir, shift=None)
        get_evals(data, trained_path, savedir, int(A))




    #data = whiten_bandpass_resample('1256663958', '1256665000', savedir)
    #get_evals(data, trained_path, savedir, 1256663958)
    # segments = np.load(args.valid_segments)
    # for valid_segment in segments:

    #     # perform full evaluation
    #     whiten_bandpass_resample(valid_segment[0], valid_segment[1], savedir)


if __name__ == '__main__':
    if 0:
        parser = argparse.ArgumentParser()
        
        # Required arguments
        parser.add_argument('--valid-segments', type=str, default=None,
                            help='File with valid segments')

        args = parser.parse_args()
        main(args)
    main()