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
    )

TIMESLIDE_TOTAL_DURATION = 3600 # 100 * 24 * 3600

##### timing eval
print(f'Time to import modules: {(time.time() - startTime_1):.2f} sec')

def extract_chunks(time_series, important_points, window_size=2048):
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

        chunks[:, idx, extracted_start:extracted_end] = time_series[:, start:end]

    return chunks

def main(args):

    device_str = 'cpu' # f'cuda:{args.gpu}' if type(args.gpu)==int else args.gpu
    gwak_models = load_gwak_models(args.model_path, DEVICE)
    startTime_2 = time.time()
    
    data = np.load(args.data_path[0])
    data = torch.from_numpy(data).to(DEVICE)
  
    ##### timing eval
    print(f'Time to load data: {(time.time() - startTime_2):.2f} sec')

    reduction = 10  # for things to fit into memory nicely

    sample_length = data.shape[1] / SAMPLE_RATE
    n_timeslides = int(TIMESLIDE_TOTAL_DURATION //
                       sample_length) * reduction
    print(f'N timeslides = {n_timeslides}, sample length = {sample_length}')
    print('Number of timeslides:', n_timeslides)

    for timeslide_num in range(1, n_timeslides + 1):
        # print(f'starting timeslide: {timeslide_num}/{n_timeslides}')

        ##### timing
        startTime_0 = time.time()

        indicies_to_slide = np.random.uniform(
            SAMPLE_RATE, data.shape[1] - SAMPLE_RATE)
        indicies_to_slide = int(indicies_to_slide)
        timeslide = torch.empty(data.shape, device=DEVICE)

        # hanford unchanged
        timeslide[0, :] = data[0, :]

        # livingston slid
        timeslide[1, :indicies_to_slide] = data[1, -indicies_to_slide:]
        timeslide[1, indicies_to_slide:] = data[1, :-indicies_to_slide]

        # make a random cut with the reduced shape
        reduced_len = int(data.shape[1] / reduction)
        start_point = int(np.random.uniform(
            0, data.shape[1] - SAMPLE_RATE - reduced_len))
        timeslide = timeslide[:, start_point:start_point + reduced_len]

        timeslide = timeslide[:, :(timeslide.shape[1] // 1000) * 1000]
        if timeslide_num==1: print(f'Time to prepare for eval {timeslide_num}/{n_timeslides} timeslide: {(time.time() - startTime_0):.2f} sec')
        startTime_01 = time.time()


        final_values = full_evaluation(
            timeslide[None, :, :], args.model_path, DEVICE, loaded_models=gwak_models)
        # print(final_values.shape)
        # print('saving, individually')
        means, stds = torch.mean(
            final_values, axis=-2), torch.std(final_values, axis=-2)
        means, stds = means.detach().cpu().numpy(), stds.detach().cpu().numpy()
        np.save(f'{args.save_normalizations_path}/normalization_params_{timeslide_num}.npy', np.stack([means, stds], axis=0))
        final_values = final_values.detach().cpu().numpy()

        if timeslide_num==1: print(f'Time to do gwak eval {timeslide_num}/{n_timeslides} timeslide: {(time.time() - startTime_01):.2f} sec')
        startTime_02 = time.time()

         save_full_timeslide_readout = True
         if save_full_timeslide_readout: 
            FAR_2days = -1.617 #lowest FAR bin we want to worry about

            #load models
            norm_factors = np.load(f"/home/katya.govorkova/gwak-paper-final-models/trained/norm_factor_params.npy")
            fm_model_path = (f"/home/katya.govorkova/gwak-paper-final-models/trained/fm_model.pt")
            fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
            fm_model.load_state_dict(torch.load(
                fm_model_path, map_location=GPU_NAME))#map_location=f'cuda:{args.gpu}'))

            #extract weights and bias
            linear_weights = fm_model.layer.weight.detach().cpu().numpy()
            bias_value = fm_model.layer.bias.detach().cpu().numpy()

            # Inference to save scores (final metric) and scaled_evals (GWAK space * weights unsummed)
            scores = []
            scaled_evals = []
            for elem in final_values[0]:

                # delete factors not used for FM
                elem = np.delete(elem, FACTORS_NOT_USED_FOR_FM, -1)

                # normalize
                elem = (elem - norm_factors[0]) / norm_factors[1]

                # multiply by weights (NOT SUMMED)
                scaled_eval = np.multiply(elem, linear_weights)
                scaled_evals.append(scaled_eval[0, :])

                # evaluate final metric (model carries bias)
                #elem = torch.from_numpy(elem).to(DEVICE)
                #scores.append(fm_model(elem).detach().cpu().numpy())
                scores.append(np.sum(scaled_eval) + bias_value) #quicker to not load gpu


            scores = np.array(scores)
            scaled_evals = np.array(scaled_evals)

            # do smoothing on the scores
            kernel_len = 50
            kernel = np.ones(kernel_len)/kernel_len
            #kernel_evals = np.ones((kernel_len, scaled_evals[0].shape[1]))/kernel_len
            bottom_trim = kernel_len * 5
            top_trim = - bottom_trim
            scores = np.apply_along_axis(lambda m : np.convolve(m, kernel, mode='same')[bottom_trim:top_trim], axis=0, arr=scores)
            scaled_evals = np.apply_along_axis(lambda m : np.convolve(m, kernel, mode='same')[bottom_trim:top_trim], axis=0, arr=scaled_evals)

            # score = (n_windows, 1) -> (n_windows, 1)
            # scaled_evals = (n_windows, 16) -> (n_windows, 16)

            indices = np.where(scores < FAR_2days)[0]   
            filtered_final_score = scores[indices]
            filtered_final_scaled_evals = scaled_evals[indices]
            timeslide = timeslide.detach().cpu().numpy()

            #extract important timeslides with indices
            important_timeslide = extract_chunks(timeslide, midpoints[indices], window_size=1024)

            #print(indices, filtered_final_score)
            #print("timeslide shape", important_timeslide.shape)
            #print("filtered_final_score", filtered_final_score.shape)
            #print("scaled_evals", filtered_final_scaled_evals.shape)

            if len(indices) > 0:
                np.savez(f'{args.save_evals_path}/timeslide_evals_FULL_{timeslide_num}.npz',
                                            final_scaled_evals=filtered_final_scaled_evals,
                                            metric_score = filtered_final_score,
                                            timeslide_data = important_timeslide,
                                            time_event_ocurred = midpoints[indices])
                    
        # save as a numpy file, with the index of timeslide_num
        np.save(f'{args.save_evals_path}/timeslide_evals_{timeslide_num}.npy', final_values)

        ##### timing eval
        if timeslide_num==1: print(f'Time to eval {timeslide_num}/{n_timeslides} timeslides: {(time.time() - startTime_0):.2f} sec')

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
