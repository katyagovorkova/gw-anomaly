import os
import sys
import argparse
import numpy as np

import torch
from torch.nn.functional import conv1d

from models import LinearModel
from evaluate_data import full_evaluation
from helper_functions import load_gwak_models

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SAMPLE_RATE,
    FACTORS_NOT_USED_FOR_FM,
    SEGMENT_OVERLAP
    )


def event_clustering(indices, scores, spacing, device):
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

    device_str = f'cuda:{args.gpu}'
    DEVICE = torch.device(device_str)
    gwak_models = load_gwak_models(args.model_path, DEVICE, device_str)

    orig_kernel = 50
    kernel_len = int(orig_kernel * 5/SEGMENT_OVERLAP)
    kernel = torch.ones((1, kernel_len)).float().to(DEVICE)/kernel_len
    kernel = kernel[None, :, :]

    data = np.load(args.data_path)
    data = torch.from_numpy(data).to(DEVICE)

    # load FM model
    # norm_factors = np.load(f"/home/katya.govorkova/gwak-paper-final-models/trained/norm_factor_params.npy")
    norm_factors = np.array([[1.4951140e+03, 1.0104435e+03, 2.1687556e+03, 6.4572485e+02,
        8.2891174e+02, 2.1687556e+03, 1.6633119e+02, 2.3331506e+02,
        2.1687556e+03, 6.6346790e+02, 9.0009998e+02, 2.1687556e+03,
        3.3232565e+02, 4.5468460e+02, 2.1687556e+03, 1.8892123e-01],
       [5.0531479e+02, 4.4439362e+02, 1.1223564e+03, 4.8320212e+02,
        5.7444623e+02, 1.1223564e+03, 2.8041806e+02, 3.8093832e+02,
        1.1223564e+03, 4.3112857e+02, 6.1296509e+02, 1.1223564e+03,
        2.1180432e+02, 3.0003491e+02, 1.1223564e+03, 3.8881097e-02]])

    fm_model_path = ("/home/katya.govorkova/gwak-paper-final-models/trained/fm_model.pt")
    fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
    fm_model.load_state_dict(torch.load(
        fm_model_path, map_location=device_str))
    mean_norm = torch.from_numpy(norm_factors[0]).to(DEVICE)#[:-1]
    std_norm = torch.from_numpy(norm_factors[1]).to(DEVICE)#[:-1]

    # extract weights and bias
    linear_weights = fm_model.layer.weight
    bias_value = fm_model.layer.bias

    reduction = 2  # for things to fit into memory nicely
    sample_length = data.shape[1] / SAMPLE_RATE
    n_timeslides = int(args.timeslide_total_duration //
                       sample_length) * reduction
    reduced_len = int(data.shape[1] / reduction)
    reduced_len = (reduced_len // 1000) * 1000
    timeslide = torch.empty((2, reduced_len)).to(DEVICE)

    for timeslide_num in range(1, n_timeslides + 1):

        # pick a random point in hanford, and one in livingston
        # bound it so don't have wrap around effect, which is okay

        hanford_start = int(np.random.uniform(0, data.shape[1]-SAMPLE_RATE-reduced_len))
        livingston_start = hanford_start
        while abs(hanford_start-livingston_start) < 41 * 5: # maximum time of flight, with extra factor
            livingston_start = int(np.random.uniform(0, data.shape[1]-SAMPLE_RATE-reduced_len))

        timeslide[0, :] = data[0, hanford_start:hanford_start+reduced_len]
        timeslide[1, :] = data[1, livingston_start:livingston_start+reduced_len]

        final_values, midpoints = full_evaluation(
                timeslide[None, :, :], args.model_path, DEVICE,
                return_midpoints=True, loaded_models=gwak_models)

        final_values = final_values[0]

        save_full_timeslide_readout = True
        if save_full_timeslide_readout:

            FAR_2days = -1.617 # lowest FAR bin we want to worry about

            # Inference to save scores (final metric) and scaled_evals (GWAK space * weights unsummed)
            final_values_slx = (final_values - mean_norm)/std_norm

            scaled_evals = torch.multiply(final_values_slx, linear_weights[None, :])[0, :]
            scores = (scaled_evals.sum(axis=1) + bias_value)[:, None]
            scaled_evals = conv1d(scaled_evals.transpose(0, 1).float()[:, None, :],
                kernel, padding="same").transpose(0, 1)[0].transpose(0, 1)
            smoothed_scores = conv1d(scores.transpose(0, 1).float()[:, None, :],
                kernel, padding="same").transpose(0, 1)[0].transpose(0, 1)

            indices = torch.where(smoothed_scores < FAR_2days)[0]

            if len(indices) > 0:
                indices = event_clustering(indices, smoothed_scores, 5*SAMPLE_RATE/SEGMENT_OVERLAP, DEVICE) # 5 seconds
            filtered_final_score = smoothed_scores.index_select(0, indices)
            filtered_final_scaled_evals = scaled_evals.index_select(0, indices)

            indices_ = indices.detach().cpu().numpy()

            # extract important timeslides with indices
            important_timeslide = extract_chunks(timeslide,
                                                midpoints[indices_],
                                                DEVICE, window_size=1024)
            filtered_final_scaled_evals = filtered_final_scaled_evals.detach().cpu().numpy()
            filtered_final_score = filtered_final_score.detach().cpu().numpy()
            if len(indices_) > 0:
                print(len(indices_))
                np.savez(f'{args.save_evals_path}/timeslide_evals_full_{timeslide_num}.npz',
                                            final_scaled_evals=filtered_final_scaled_evals,
                                            metric_score = filtered_final_score,
                                            timeslide_data = important_timeslide,
                                            time_event_ocurred = midpoints[indices_])
        if not os.path.exists(f"{args.save_evals_path}/livetime_tracker.npy"):
            track_data = np.zeros(1)
            np.save(f"{args.save_evals_path}/livetime_tracker.npy", track_data)
        tracked_sofar = np.load(f"{args.save_evals_path}/livetime_tracker.npy")
        np.save(f"{args.save_evals_path}/livetime_tracker.npy", tracked_sofar + reduced_len/SAMPLE_RATE)

        # save as a numpy file, with the index of timeslide_num
        np.save(f'{args.save_evals_path}/timeslide_evals_{timeslide_num}.npy', final_values.detach().cpu().numpy())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('model_path', nargs='+', type=str,
                        help='Path to the models')

    # Additional arguments
    parser.add_argument('--data-path', type=str,
                        help='File containing the background to create timeslides from')

    parser.add_argument('--timeslide-total-duration', type=int,
                        help='How long of timeslides to create')

    parser.add_argument('--files-to-eval', type=int, default=1,
                        help='How many 1-hour background fiels to turn into timeslides')

    parser.add_argument('--gpu', type=str, default='1',
                        help='On which GPU to run')

    parser.add_argument('--save-evals-path', type=str, default=None,
                        help='Where to save evals')

    args = parser.parse_args()

    folder_path = args.data_path
    save_evals_folder = args.save_evals_path
    for i, filename in enumerate(os.listdir(folder_path)):

        if i >= args.files_to_eval and args.files_to_eval!=-1:
            break

        if filename.endswith('.npy'):

            args.data_path = os.path.join(folder_path, filename)
            args.save_evals_path = f"{save_evals_folder}/{filename[:-4]}/"
            os.makedirs(args.save_evals_path, exist_ok=True)
            main(args)
            print(f'Finished running on {filename}')
