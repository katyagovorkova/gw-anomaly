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
from config import (
    CHANNEL,
    GPU_NAME,
    SEGMENT_OVERLAP,
    SAMPLE_RATE,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    FACTORS_NOT_USED_FOR_FM
    )
from helper_functions import far_to_metric, compute_fars
DEVICE = torch.device(GPU_NAME)


def clustering(x, bar=5*4096/5):

    # 5 second spacing between events

    cluster = []
    cluster.append(x[0])

    for i, point in enumerate(x[1:]):
        if x[i] - x[i-1] > bar:
            cluster.append(point)

    return cluster


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

def get_evals(data, trained_path, savedir, start_point, extra = "", far_search=None):
    supervised_model = False
    if trained_path[:-1].split("/")[-1]=="O3av2_non_linear_bbh_only":
        supervised_model = True
    strain = np.stack(data, axis=1)[np.newaxis, :, :]
    strain_orig = np.stack(data, axis=1)
    strain = np.swapaxes(strain, 1, 2) # make it (N_batches, 2, time_axis), N_batches is 1 here

    #break the strain into pieces along the time axis to fit into GPU memory
    max_gpu_seconds =350 # 500 seconds at a time
    max_gpu_dtps = max_gpu_seconds * SAMPLE_RATE

    n_splits = (strain.shape[2] // max_gpu_dtps) + 1 
    split_data = []
    for n in range(n_splits):
        split_data.append(strain[:, :, n*max_gpu_dtps:(n+1)*max_gpu_dtps])


    model_path = f"{trained_path}/trained/models/"
    model_paths = []
    for elem in ["background.pt", "bbh.pt",  "glitches.pt", 
                 "sghf.pt", "sglf.pt"]:
        model_paths.append(f'{model_path}/{elem}')

    #if not supervised_model:
    norm_factors = np.load(f"{trained_path}/trained/norm_factor_params.npy")

    device=DEVICE
    evals = []
    midpoints = []
    for i, strain_elem in enumerate(split_data):
        ev, mdp = full_evaluation(strain_elem, model_paths, device, return_midpoints=True)
        ev = ev.cpu().detach().numpy()
        ev = np.delete(ev, FACTORS_NOT_USED_FOR_FM, -1)
        evals.append(ev)
        midpoints.append(mdp+i*(max_gpu_dtps))

    
    fm_model_path = (f"{trained_path}/trained/fm_model.pt")

    fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM), override_supervised=supervised_model).to(DEVICE)
    fm_model.load_state_dict(torch.load(
        fm_model_path, map_location=GPU_NAME))#map_location=f'cuda:{args.gpu}'))
    if not supervised_model:
        linear_weights = fm_model.layer.weight.detach().cpu().numpy()
        bias_value = fm_model.layer.bias.detach().cpu().numpy()

    scores = []
    scaled_evals = []
    for elem in evals:
        elem = (elem - norm_factors[0]) / norm_factors[1]
        if not supervised_model:
            scaled_eval = np.multiply(elem, linear_weights)
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
    if not supervised_model:
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
    
    #print(scaled_evals.shape)
    if not supervised_model:
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


    if not supervised_model:
        far_bins = np.load(f"/home/katya.govorkova/gwak-paper-final-models/far_bins_{kernel_len}.npy")
        # set a bar at 1/2 days FAR to get elements of interest
    else:
        far_bins = np.load(f"/home/katya.govorkova/gw-anomaly/output/O3av2_non_linear_bbh_only/far_bins_k{kernel_len}.npy")

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
    for j in range(len(midpoint_clusters)):
        loudest = eval_locations[j]
        #print("loudest", loudest)

        left_edge = 150
        right_edge = 225
        n_points = left_edge+right_edge
        labels = ['background', 'bbh', 'glitch', 'sglf', 'sghf', 'pearson']
        quak_evals_ts = np.linspace(0, n_points*(SEGMENT_OVERLAP/SAMPLE_RATE), n_points)
        #if len(evals[loudest-left_edge:loudest+right_edge, i]) != len(quak_evals_ts):
        #    continue #bypassing edge effects

        

        #get rid of edge effects
        if loudest < 200 or loudest > len(scores) - 201:
            continue
        try:
            best_score = min(scores[loudest-left_edge:loudest+right_edge])
        except ValueError:
            continue

        fig, axs = plt.subplots(2, 2, figsize=(28, 17))

        if not supervised_model:
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
        axs[1, 0].plot(quak_evals_ts*1000, scores[loudest-left_edge:loudest+right_edge]-bias_value, label = 'final metric', c='black')
        axs[1, 0].plot([], [], '-', label="Hanford", c="black")
        axs[1, 0].plot([], [], '--', label="Livingston", c="black")
        axs[1, 0].legend(handlelength=3, fontsize=17)
        axs[1, 0].set_xlabel("Time, (ms)")
        axs[1, 0].set_ylabel("Final Metric Contribution")
        
        #axs[0].legend()
        #axs[0].set_xlabel('Time, (ms)')
        #axs[0].set_ylabel('Contribution to final metric')

        p = all_midpoints[loudest]
        left_edge, right_edge = left_edge * SEGMENT_OVERLAP, right_edge*SEGMENT_OVERLAP
        strain_ts = np.linspace(0, (right_edge+left_edge)/SAMPLE_RATE, right_edge+left_edge)
        #print("190", strain_orig.shape, p-left_edge,p+right_edge)
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
        
        
        #print(np.array(H_hq))
        #print(np.array(H_hq).shape)
        #print(H_hq.shape)
        #axs[0, 1].imshow()
        #assert 0
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
        # = compute_fars(np.array([best_score]), far_bins)[0]
        #plt.tight_layout()
        plt.savefig(f'{savedir}/{start_point}/{start_point+p/SAMPLE_RATE:.3f}_{best_far}_{best_score:.2f}{extra}.png', dpi=300, bbox_inches="tight")
        plt.close()

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

def IGNORE_main(args):
    trained_path = "/home/katya.govorkova/gwak-paper-final-models/"

    savedir = 'output/NAME/'
    savedir = None
    A, B = None, None # start and stop of gps time

    #fill it in (or alternative data loading)
    assert A is not None and B is not None and savedir is not None

    data = whiten_bandpass_resample(A, B, savedir, shift=None)
    # data is [strain_H1, strain_L1], where each is downsampled, whitened, and bandpassed
    get_evals(data, trained_path, savedir, int(A))

def main(args):
    gwtc_events=parse_gwtc_catalog("/home/ryan.raikman/s22/forks/katya/gw-anomaly/data/gwtc.csv", 
                                  1238166018, 1253977218)

    valid_segments = np.load("/home/katya.govorkova/gwak-paper-final-models/O3a_intersections.npy")
    trained_path = "/home/katya.govorkova/gwak-paper-final-models/" # fix hardcoding later
    #trained_path = "/home/katya.govorkova/gw-anomaly/output/O3av2_non_linear_bbh_only/"
    savedir = 'output/all_O3a_spectrogram/'

    far_a, far_b = make_metric_vs_far(savedir)


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
    if 1:
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

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--valid-segments', type=str, default=None,
                        help='File with valid segments')

    args = parser.parse_args()
    main(args)