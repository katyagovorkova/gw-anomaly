import os
import sys
import time
import argparse
import numpy as np

import torch
from torch.nn.functional import conv1d

from models import LinearModel
from evaluate_data import full_evaluation
from helper_functions import load_gwak_models, split_into_segments_torch, std_normalizer_torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SAMPLE_RATE,
    FACTORS_NOT_USED_FOR_FM,
    SEGMENT_OVERLAP,
    SEG_NUM_TIMESTEPS
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

def create_ffts_by_detec(data, device):
    clipped_time_axis = (data.shape[2] // SEGMENT_OVERLAP) * SEGMENT_OVERLAP
    data = data[:, :, :clipped_time_axis]

    segments = split_into_segments_torch(data, device=device)

    segments_normalized = std_normalizer_torch(segments)

    # segments_normalized at this point is (N_batches, N_samples, 2, 100) and
    # must be reshaped into (N_batches * N_samples, 2, 100) to work with
    # quak_predictions
    N_batches, N_samples = segments_normalized.shape[
        0], segments_normalized.shape[1]
    segments_normalized = torch.reshape(
        segments_normalized, (N_batches * N_samples, 2, SEG_NUM_TIMESTEPS))

    H_ffts = torch.fft.rfft(segments_normalized[:, 0, :], axis=-1)
    L_ffts = torch.fft.rfft(segments_normalized[:, 1, :], axis=-1)
    return H_ffts, L_ffts

def main(args):

    device_str = f'cuda:{args.gpu}'
    #device_str = "cpu"
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
    freq_mean_norm = mean_norm[2]
    freq_std_norm = std_norm[2]

    #extract weights and bias
    linear_weights = fm_model.layer.weight#.detach().cpu().numpy()
    bias_value = fm_model.layer.bias#.detach().cpu().numpy()
    linear_weights = linear_weights#[:, :-1]

    # extract weights and bias
    linear_weights = fm_model.layer.weight
    bias_value = fm_model.layer.bias

    #reduction = 1  # for things to fit into memory nicely

    sample_length = data.shape[1] / SAMPLE_RATE
    n_timeslides = int(args.timeslide_total_duration //
                       sample_length)# * reduction
    # print(f'N timeslides = {n_timeslides}, sample length = {sample_length}')
    # print('Number of timeslides:', n_timeslides)
    print(f'N timeslides = {n_timeslides}, sample length = {sample_length}')
    print('Number of timeslides:', n_timeslides)

    timeslide = torch.empty(data.shape, device=DEVICE)
    reduced_len = (data.shape[1] // 1000) * 1000
    timeslide = torch.empty((2, reduced_len)).to(DEVICE)

    data = data[:, :reduced_len]
    timeslide = data
    tfft = time.time()
    H_ffts, L_ffts = create_ffts_by_detec(data[None, :, :], DEVICE)
    print("fft time", time.time()-tfft)
    # shift by the maximum possible correlation to start (more)
    decorrelate = 50
    timeslide[1, :] = torch.roll(timeslide[1, :], SEGMENT_OVERLAP*decorrelate)
    L_ffts = torch.roll(L_ffts, decorrelate)

    for timeslide_num in range(1, n_timeslides + 1):

        # pick a random point in hanford, and one in livingston
        # bound it so don't have wrap around effect, which is okay
        if 0:
            hanford_start = int(np.random.uniform(0, data.shape[1]-SAMPLE_RATE-reduced_len))
            livingston_start = hanford_start
            while abs(hanford_start-livingston_start) < 41 * 5: # maximum time of flight, with extra factor
                livingston_start = int(np.random.uniform(0, data.shape[1]-SAMPLE_RATE-reduced_len))

            timeslide[0, :] = data[0, hanford_start:hanford_start+reduced_len]
            timeslide[1, :] = data[1, livingston_start:livingston_start+reduced_len]
        else:

            timeslide[1, :] = torch.roll(timeslide[1, :], SEGMENT_OVERLAP) #keep doing it
            L_ffts = torch.roll(L_ffts, 1)

        # # compute the filter based on freq_corr cut
        # dot =  (torch.abs(torch.linalg.vecdot(
        #     H_ffts, L_ffts, axis=-1)) - freq_mean_norm) / freq_std_norm
        # # rescale it to make the cut


        # freq_corr_weight = -0.54
        # bar = -1 * freq_corr_weight

        # selection = torch.where(dot>bar)[0]
        # print(f"size reduction: {len(selection)}") #/H_ffts.shape[0]:.2f

        final_values, midpoints = full_evaluation(
                timeslide[None, :, :], args.model_path, DEVICE,
                return_midpoints=True, loaded_models=gwak_models,
                # selection=selection)
                selection=None)
        if 0:
            means, stds = torch.mean(
                final_values, axis=-2), torch.std(final_values, axis=-2)
            means, stds = means.detach().cpu().numpy(), stds.detach().cpu().numpy()
            np.save(f'{args.save_normalizations_path}/normalization_params_{timeslide_num}.npy', np.stack([means, stds], axis=0))
        #if timeslide_num>0: print(f'Time to do gwak eval {timeslide_num}/{n_timeslides} timeslide: {(time.time() - startTime_01):.2f} sec')
        #startTime_02 = time.time()
        final_values = final_values[0]

        save_full_timeslide_readout = False
        if save_full_timeslide_readout:

            FAR_2days = -1.617 # lowest FAR bin we want to worry about

            # Inference to save scores (final metric) and scaled_evals (GWAK space * weights unsummed)
            final_values_slx = (final_values - mean_norm)/std_norm

            scaled_evals = torch.multiply(final_values_slx, linear_weights[None, :])[0, :]
            print("scaled_evals", scaled_evals.shape)
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
    print(folder_path)
    #p = np.random.permutation(len(os.listdir(folder_path)))

    print("N files", args.files_to_eval)
    save_evals_path = args.save_evals_path
    for i, filename in enumerate(np.array(os.listdir(folder_path))):#[p]):

        if i >= args.files_to_eval and args.files_to_eval != -1:
            break

        if filename.endswith('.npy'):

            args.data_path = os.path.join(folder_path, filename)
            args.save_evals_path = f"{save_evals_path}/{filename[:-4]}/"
            os.makedirs(args.save_evals_path, exist_ok=True)
            main(args)
            print(f'Finished running on {filename}')
