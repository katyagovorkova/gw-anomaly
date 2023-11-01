import time

##### timing
startTime_1 = time.time()

import os
import argparse
import numpy as np

import torch
import torch.nn as nn

from models import LinearModel
from evaluate_data import full_evaluation
from helper_functions import load_gwak_models
import sys
import time
from torch.nn.functional import conv1d

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    TIMESLIDE_TOTAL_DURATION,
    SAMPLE_RATE,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN,
    RETURN_INDIV_LOSSES,
    FACTORS_NOT_USED_FOR_FM,
    SMOOTHING_KERNEL_SIZES,
    DO_SMOOTHING,
    GPU_NAME
    )

TIMESLIDE_TOTAL_DURATION = 3600 # 100 * 24 * 3600

##### timing eval
print(f'Time to import modules: {(time.time() - startTime_1):.2f} sec')

def extract_chunks(time_series, important_points, device, window_size=2048):
    # Determine the dimensions of the output array
    n_chunks = len(important_points)
    chunk_length = 2 * window_size + 1

    # Initialize an empty array to store the chunks
    chunks = np.zeros((2, n_chunks, chunk_length))
    for idx, point in enumerate(important_points):
        # Define the start and end points for extraction
        start = max(0, point - window_size)
        end = min(time_series.shape[1], point + window_size + 1)

        # Handle edge cases
        extracted_start = window_size - (point - start)
        extracted_end = extracted_start + (end - start)

        #chunks[:, idx, extracted_start:extracted_end] = time_series[:, start:end]
        idxs = torch.arange(start, end, 1).to(device)
        chunks[:, idx, extracted_start:extracted_end] = time_series.index_select(1,idxs).detach().cpu().numpy()

    return chunks

def main(args):

    #device_str = 'cpu' 
    device_str = f'cuda:{args.gpu}'# if type(args.gpu)==int else args.gpu
    DEVICE = torch.device(device_str)
    gwak_models = load_gwak_models(args.model_path, DEVICE, device_str)
    startTime_2 = time.time()

    #manual_pearson_values = torch.zeros(([1, 293945])).to(DEVICE)
    kernel_len = 50
    kernel = torch.ones((1, kernel_len)).float().to(DEVICE)/kernel_len
    kernel = kernel[None, :, :]

    data = np.load(args.data_path[0])
    data = torch.from_numpy(data).to(DEVICE)
    factors_used_for_fm = np.linspace(0, 20, 21)

    #load models
    norm_factors = np.load(f"/home/katya.govorkova/gwak-paper-final-models/trained/norm_factor_params.npy")
    fm_model_path = (f"/home/katya.govorkova/gwak-paper-final-models/trained/fm_model.pt")
    fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
    fm_model.load_state_dict(torch.load(
        fm_model_path, map_location=device_str))#map_location=f'cuda:{args.gpu}'))
    mean_norm = torch.from_numpy(norm_factors[0]).to(DEVICE)#[:-1]
    std_norm = torch.from_numpy(norm_factors[1]).to(DEVICE)#[:-1]
    
    #extract weights and bias
    linear_weights = fm_model.layer.weight#.detach().cpu().numpy()
    bias_value = fm_model.layer.bias#.detach().cpu().numpy()
    linear_weights = linear_weights#[:, :-1]

    #factors_used_for_fm[-1] = -1
    for elem in FACTORS_NOT_USED_FOR_FM:
        #factors_used_for_fm.delete(elem)
        factors_used_for_fm[elem] = -1

    factors_used_for_fm = factors_used_for_fm[factors_used_for_fm>=0]
    factors_used_for_fm = torch.from_numpy(factors_used_for_fm).to(DEVICE).int()
    ##### timing eval
    print(f'Time to load data: {(time.time() - startTime_2):.2f} sec')

    reduction = 8  # for things to fit into memory nicely

    sample_length = data.shape[1] / SAMPLE_RATE
    n_timeslides = int(TIMESLIDE_TOTAL_DURATION //
                       sample_length) * reduction
    print(f'N timeslides = {n_timeslides}, sample length = {sample_length}')
    print('Number of timeslides:', n_timeslides)

    #indices = torch.zeros([367545]).to(DEVICE)
    
    startTime_1 = time.time()
    timeslide = torch.empty(data.shape, device=DEVICE)
    reduced_len = int(data.shape[1] / reduction)
    reduced_len = (reduced_len // 1000) * 1000
    timeslide = torch.empty((2, reduced_len)).to(DEVICE)

    for timeslide_num in range(1, n_timeslides + 1):
        ##### timing    
        startTime_0 = time.time()

        # pick a random point in hanford, and one in livingston
        # bound it so don't have wrap around effect, which is okay 

        hanford_start = int(np.random.uniform(0, data.shape[1]-SAMPLE_RATE-reduced_len))
        livingston_start = hanford_start
        while abs(hanford_start-livingston_start) < 41 * 5: # maximum time of flight, with extra factor
            livingston_start = int(np.random.uniform(0, data.shape[1]-SAMPLE_RATE-reduced_len))

        timeslide[0, :] = data[0, hanford_start:hanford_start+reduced_len]
        timeslide[1, :] = data[1, livingston_start:livingston_start+reduced_len]

        final_values, midpoints = full_evaluation(
                timeslide[None, :, :], args.model_path, DEVICE, return_midpoints=True, loaded_models=gwak_models)
        if 0:
            means, stds = torch.mean(
                final_values, axis=-2), torch.std(final_values, axis=-2)
            means, stds = means.detach().cpu().numpy(), stds.detach().cpu().numpy()
            np.save(f'{args.save_normalizations_path}/normalization_params_{timeslide_num}.npy', np.stack([means, stds], axis=0))
        #if timeslide_num>0: print(f'Time to do gwak eval {timeslide_num}/{n_timeslides} timeslide: {(time.time() - startTime_01):.2f} sec')
        startTime_02 = time.time()
        final_values = final_values[0]

        save_full_timeslide_readout = True
        if save_full_timeslide_readout: 
            FAR_2days = -1.617 #lowest FAR bin we want to worry about

            # Inference to save scores (final metric) and scaled_evals (GWAK space * weights unsummed)
            final_values_slx = final_values.index_select(1, factors_used_for_fm)
            final_values_slx = (final_values_slx - mean_norm)/std_norm

            scaled_evals = torch.multiply(final_values_slx, linear_weights[None, :])[0, :]
            scores = (scaled_evals.sum(axis=1) + bias_value)[:, None]
            scaled_evals = conv1d(scaled_evals.transpose(0, 1).float()[:, None, :], kernel, padding = "same").transpose(0, 1)[0].transpose(0, 1)
            smoothed_scores = conv1d(scores.transpose(0, 1).float()[:, None, :], kernel, padding = "same").transpose(0, 1)[0].transpose(0, 1)
            
            indices = torch.where(smoothed_scores < FAR_2days)[0]
            filtered_final_score = smoothed_scores.index_select(0, indices)
            filtered_final_scaled_evals = scaled_evals.index_select(0, indices)

            indices_ = indices.detach().cpu().numpy()
            #extract important timeslides with indices
            important_timeslide = extract_chunks(timeslide, 
                                                midpoints[indices_], 
                                                DEVICE, window_size=1024)
            filtered_final_scaled_evals = filtered_final_scaled_evals.detach().cpu().numpy()
            filtered_final_score = filtered_final_score.detach().cpu().numpy()
            if len(indices_) > 0:
                np.savez(f'{args.save_evals_path}/timeslide_evals_FULL_{timeslide_num}.npz',
                                            final_scaled_evals=filtered_final_scaled_evals,
                                            metric_score = filtered_final_score,
                                            timeslide_data = important_timeslide,
                                            time_event_ocurred = midpoints[indices_])

        # save as a numpy file, with the index of timeslide_num
        #np.save(f'{args.save_evals_path}/timeslide_evals_{timeslide_num}.npy', final_values.detach().cpu().numpy())
        if timeslide_num>0: print(f'Time to compute FM {timeslide_num}/{n_timeslides} timeslides: {(time.time() - startTime_02):.2f} sec')

        ##### timing eval
        if timeslide_num>0: print(f'Time to eval {timeslide_num}/{n_timeslides} timeslides: {(time.time() - startTime_0):.2f} sec')

    print(f'Time to eval all {n_timeslides} timeslides: {(time.time() - startTime_1):.2f} sec')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('model_path', nargs='+', type=str,
                        help='Path to the models')

    # Additional arguments
    parser.add_argument('--data-path', type=str, nargs='+',
                        help='File containing the background to create timeslides from')

    parser.add_argument('--gpu', type=str, default='1',
                        help='On which GPU to run')

    parser.add_argument('--save-evals-path', type=str, default=None,
                        help='Where to save evals')

    parser.add_argument('--save-normalizations-path', type=str, default=None,
                        help='Where to save normalizations')

    args = parser.parse_args()

    main(args)
