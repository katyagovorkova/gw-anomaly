import os
import numpy as np
import argparse
import torch
from gwpy.timeseries import TimeSeries
from astropy import units as u
import matplotlib.pyplot as plt

from evaluate_data import full_evaluation
from config import (
    CHANNEL,
    GPU_NAME,
    SEGMENT_OVERLAP,
    SAMPLE_RATE,
    BANDPASS_HIGH,
    BANDPASS_LOW
    )


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
        bandpass_high=BANDPASS_HIGH):

    device = torch.device(GPU_NAME)

    # Load LIGO data
    strainL1 = TimeSeries.get(f'L1:{CHANNEL}', start_point, end_point)
    strainH1 = TimeSeries.get(f'H1:{CHANNEL}', start_point, end_point)

    t0 = int(strainL1.t0 / u.s)
    print(t0)

    # Whiten, bandpass, and resample
    strainL1 = strainL1.whiten()
    strainL1 = strainL1.bandpass(bandpass_low, bandpass_high)
    strainL1 = strainL1.resample(sample_rate)

    strainH1 = strainH1.whiten()
    strainH1 = strainH1.bandpass(bandpass_low, bandpass_high)
    strainH1 = strainH1.resample(sample_rate)

    # split strainL1 and strainH1 into 3 equal-sized parts each for GPU data loading
    total_elements = strainH1.shape[0]
    split_size = total_elements // 3

    strain_H1_1 = np.array(strainH1[:split_size])
    strain_H1_2 = np.array(strainH1[split_size:2*split_size])
    strain_H1_3 = np.array(strainH1[2*split_size:])
    strain_L1_1 = np.array(strainL1[:split_size])
    strain_L1_2 = np.array(strainL1[split_size:2*split_size])
    strain_L1_3 = np.array(strainL1[2*split_size:])

    strain_H1_3 = strain_H1_3[:split_size]
    strain_L1_3 = strain_L1_3[:split_size]

    if 0:
        #save files
        np.save(savedir + '/H1_BurstBenchmark_%s.npy'%int(t0), strain_H1_1)
        np.save(savedir + '/H1_BurstBenchmark_%s.npy'%int(t0+(split_size/4096)), strain_H1_2)
        np.save(savedir + '/H1_BurstBenchmark_%s.npy'%int(t0+(2*split_size/4096)), strain_H1_3)
        np.save(savedir + '/L1_BurstBenchmark_%s.npy'%int(t0), strain_L1_1)
        np.save(savedir + '/L1_BurstBenchmark_%s.npy'%int(t0+(split_size/4096)), strain_L1_2)
        np.save(savedir + '/L1_BurstBenchmark_%s.npy'%int(t0+(2*split_size/4096)), strain_L1_3)

    strain_H1 = np.stack([strain_H1_1, strain_H1_2, strain_H1_3], axis=0)
    strain_L1 = np.stack([strain_L1_1, strain_L1_2, strain_L1_3], axis=0)

    strain = np.stack([strain_H1, strain_L1], axis=1)
    model_path = 'output/trained/models/'
    model_paths = []
    for elem in os.listdir(model_path):
        model_paths.append(f'{model_path}/{elem}')
    evals = []
    midpoints = []
    for i in range(3):
        ev, midp = full_evaluation(strain[i:i+1], model_paths, device, return_midpoints=True)
        evals.append(ev.detach().cpu().numpy())
        midpoints.append(midp + split_size*i)


    midpoints = np.stack(midpoints)
    midpoints = np.hstack(midpoints)
    evals = np.vstack(np.stack(evals, axis=1)[0])
    params = np.load('output/trained/final_metric_params.npy')
    means, stds = np.load('output/trained/norm_factor_params.npy')
    evals = (evals-means)/stds
    final = np.dot(evals, params)
    evals = np.multiply(evals, params) #for plotting purposes
    strains = np.stack([np.array(strainH1), np.array(strainL1)])
    if 0:
        np.save('./evals.npy', evals)
        np.save('./final.npy', final)
        np.save('./strains.npy', strains)
        np.save('./midpoints.npy', midpoints)

    final_metric_bar = -11 # corresponds rougly to 1e-5 or 1/2 days, as they describe it
    if len(midpoints[final<final_metric_bar]) == 0: return None  # no significant signal found
    midpoint_clusters = clustering(midpoints[final<final_metric_bar])
    eval_locations = []
    for point in midpoint_clusters:
        eval_locations.append(np.where(midpoints == point)[0][0])

    n_detections = len(midpoint_clusters)
    try:
        os.makedirs(f'{savedir}/{start_point}/')
    except FileExistsError:
        None
    for j in range(len(midpoint_clusters)):
        loudest = eval_locations[j]


        left_edge = 200
        right_edge = 400
        n_points = left_edge+right_edge
        labels = ['background', 'bbh', 'glitch', 'sglf', 'sghf', 'pearson']
        quak_evals_ts = np.linspace(0, n_points*(SEGMENT_OVERLAP/SAMPLE_RATE), n_points)
        if len(evals[loudest-left_edge:loudest+right_edge, i]) != len(quak_evals_ts):
            continue #bypassing edge effects

        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        for i in range(5):
            axs[0].plot(quak_evals_ts*1000, evals[loudest-left_edge:loudest+right_edge, i], label = labels[i])
        axs[0].plot(quak_evals_ts*1000, final[loudest-left_edge:loudest+right_edge], label = 'final metric')
        axs[0].legend()
        axs[0].set_xlabel('Time, (ms)')
        axs[0].set_ylabel('Contribution to final metric')

        p = midpoints[loudest]
        left_edge, right_edge = left_edge * SEGMENT_OVERLAP, right_edge*SEGMENT_OVERLAP
        strain_ts = np.linspace(0, (right_edge+left_edge)/SAMPLE_RATE, right_edge+left_edge)
        axs[1].plot(strain_ts*1000, strains[0, p-left_edge:p+right_edge], label = 'Hanford', alpha=0.8)
        axs[1].plot(strain_ts*1000, strains[1, p-left_edge:p+right_edge], label = 'Livingston', alpha=0.8)
        axs[1].set_xlabel('Time, (ms)')
        axs[1].set_ylabel('strain')
        axs[1].legend()
        axs[1].set_title(f'strain index: {p}, gps time: {p/SAMPLE_RATE:.3f} + {start_point}')



        plt.savefig(f'{savedir}/{start_point}/graphs_{j}_{p/SAMPLE_RATE:.3f}.png', dpi=300)
        plt.close()
    gps_times = []
    for j in range(len(midpoint_clusters)):
        loudest = eval_locations[j]
        p = midpoints[loudest]
        gps_times.append(p/4096 + int(start_point))
    gps_times = np.array(gps_times)

    #compile the desired info
    analysis_results = np.zeros((len(midpoint_clusters), 7))
    #peak GPS, start GPS, end GPS, characteristic_frequency, lower_frequency, upper_frequency, ranking_statistic_value, false_alarm_rate
    print('midpoint clusters', midpoint_clusters)
    for j in range(len(midpoint_clusters)):
        loudest = eval_locations[j]
        #p = midpoints[loudest]
        print('152', np.where(midpoints == midpoint_clusters[j]))
        loc = np.where(midpoints == midpoint_clusters[j])[0][0]



        search_window = int(2.5 * SAMPLE_RATE/SEGMENT_OVERLAP) # 2.5 seconds to the left, 5 seconds to the right (since 'loudest' is more or less the start time)
        search_space = np.arange(loc-search_window, loc+2*search_window, SEGMENT_OVERLAP)
        #search_space = midpoints[loudest-search_window:loudest+2*search_window]
        start = None
        lowest_midp = None
        lowest_val = 0
        end = None
        #assert final[midpoints[loudest]
        print(search_space)
        print(search_space[0], search_space[-1])
        for s in search_space:
            #print(np.where(midpoints==x))
            #s = np.where(midpoints==x)[0][0] #convert to an index for final array

            if s >= 0 and s < len(final) and final[s] < final_metric_bar: # also make sure that the considered window has not extended outside the segment
                #valid point for consideration
                if start == None:
                    start = midpoints[s]

                if final[s] < lowest_val:
                    lowest_val = final[s]
                    lowest_midp = midpoints[s]

                end = midpoints[s]

        print('start, lowest_val, lowest, end', start, lowest_val, lowest_midp, end)

        if lowest_midp != None:

            analysis_results[j, 0] = lowest_midp/4096 + int(start_point)
            analysis_results[j, 1] = start/4096 + int(start_point)
            analysis_results[j, 2] = end/4096 + int(start_point)

            analysis_results[j, 6] = lowest_val

    np.save(f'{savedir}/{start_point}/gps_times.npy', gps_times)
    np.savetxt(f'{savedir}/{start_point}/analysis_results_{start_point}_{end_point}.txt', analysis_results)


def main(args):

    savedir = 'output/O3b/'

    whiten_bandpass_resample('1256663958', '1256665000', savedir)

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