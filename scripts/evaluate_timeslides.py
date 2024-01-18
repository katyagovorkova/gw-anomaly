import os
import sys
import time
import argparse
import numpy as np
import time
import torch
from torch.nn.functional import conv1d

from models import LinearModel
from evaluate_data import full_evaluation
from helper_functions import load_gwak_models, split_into_segments_torch, std_normalizer_torch
from scipy.stats import pearsonr
from scipy.signal import welch
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SAMPLE_RATE,
    FACTORS_NOT_USED_FOR_FM,
    SEGMENT_OVERLAP,
    SEG_NUM_TIMESTEPS,
    MODELS_LOCATION
    )

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
                new[:, -2] += x[:, i]

        new[:, -1] = x[:, -1]

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

def extract_chunks(strain_data, timeslide_num, important_points, device, roll_amount = SEG_NUM_TIMESTEPS, window_size=1024):
    '''
    Important points are indicies into thestrain_data
    '''
    L_shift = timeslide_num*roll_amount
    timeslide_len = strain_data.shape[1]
    edge_check_passed = []

    fill_strains = np.zeros((len(important_points), 2, window_size*2))
    for idx, point in enumerate(important_points):
        # check that the point is not on the edge
        edge_check_passed.append(abs(point - timeslide_len)% timeslide_len > window_size*2)
        if abs(point - timeslide_len)% timeslide_len > window_size*2:
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

def main(args):

    DEVICE = torch.device(f'cuda:{args.gpu}')
    model_path = args.model_path if not args.from_saved_models else \
        [os.path.join(MODELS_LOCATION, os.path.basename(f)) for f in args.model_path]
    gwak_models = load_gwak_models(model_path, DEVICE, f'cuda:{args.gpu}')

    orig_kernel = 50
    kernel_len = int(orig_kernel * 5/SEGMENT_OVERLAP)
    kernel = torch.ones((1, kernel_len)).float().to(DEVICE)/kernel_len
    kernel = kernel[None, :, :]

    if heuristics_tests:
        long_relation = np.load("/home/ryan.raikman/share/gwak/long_relation.npy")
        short_relation = np.load("/home/ryan.raikman/share/gwak/short_relation.npy")

    gwak_models_ = load_gwak_models(args.model_path, DEVICE, device_str)
    norm_factors = np.load(f"/home/katya.govorkova/gwak-paper-final-models/trained/norm_factor_params.npy")
    # norm_factors = np.array([[1.4951140e+03, 1.0104435e+03, 2.1687556e+03, 6.4572485e+02,
    #     8.2891174e+02, 2.1687556e+03, 1.6633119e+02, 2.3331506e+02,
    #     2.1687556e+03, 6.6346790e+02, 9.0009998e+02, 2.1687556e+03,
    #     3.3232565e+02, 4.5468460e+02, 2.1687556e+03, 1.8892123e-01],
    #    [5.0531479e+02, 4.4439362e+02, 1.1223564e+03, 4.8320212e+02,
    #     5.7444623e+02, 1.1223564e+03, 2.8041806e+02, 3.8093832e+02,
    #     1.1223564e+03, 4.3112857e+02, 6.1296509e+02, 1.1223564e+03,
    #     2.1180432e+02, 3.0003491e+02, 1.1223564e+03, 3.8881097e-02]])

    fm_model_path = ("/home/katya.govorkova/gwak-paper-final-models/trained/fm_model.pt")
    fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
    fm_model.load_state_dict(torch.load(
        fm_model_path, map_location=device_str))

    linear_weights = fm_model.layer.weight#.detach().cpu().numpy()
    bias_value = fm_model.layer.bias#.detach().cpu().numpy()

    mean_norm = torch.from_numpy(norm_factors[0]).to(DEVICE)#[:-1]
    std_norm = torch.from_numpy(norm_factors[1]).to(DEVICE)#[:-1]

    roll_amount = SEG_NUM_TIMESTEPS // SEGMENT_OVERLAP
    ts = time.time()
    not_finished = True
    initial_roll = 0
    while not_finished:
        data = np.load(args.data_path)
        reduced_len = (data.shape[1] // 1000) * 1000
        data = data[:, :reduced_len]

        data = torch.from_numpy(data).to(DEVICE)
        data[1, :] = torch.roll(data[1, :], initial_roll)
        strain_data = np.copy(data.cpu().numpy())
        sample_length = data.shape[1] / SAMPLE_RATE
        n_timeslides = int(args.timeslide_total_duration //
                        sample_length)

        print(f'N timeslides = {n_timeslides}, sample length = {sample_length}')
        print('Number of timeslides:', n_timeslides)


        if not torch.is_tensor(data):
            data = torch.from_numpy(data).to(DEVICE)
        data = data[None, :, :]
        assert data.shape[1] == 2
        clipped_time_axis = (data.shape[2] // SEGMENT_OVERLAP) * SEGMENT_OVERLAP
        data = data[:, :, :clipped_time_axis]

        segments = split_into_segments_torch(data, device=DEVICE)
        segments_normalized = std_normalizer_torch(segments)

        RNN_precomputed_all, _ = full_evaluation(
                    segments_normalized, args.model_path, DEVICE, 
                    return_midpoints=True, loaded_models=None, grad_flag=False,
                    do_rnn_precomp=True, already_split=True)
        # extract the batch size
        RNN_precomputed = {}
        for key in ["bbh", "sghf", "sglf"]:
            RNN_precomputed[key] = RNN_precomputed_all[key][0]
        batch_size_ = RNN_precomputed_all['bbh'][1]

        timeslide = segments_normalized
        gwak_models = load_gwak_models(args.model_path, DEVICE, device_str, load_precomputed_RNN=True, batch_size=batch_size_)

        for timeslide_num in range(1, n_timeslides + 1):
            if timeslide_num != 1: print(f"throughput {3600/(time.time()-ts):.2f} Hz")
            ts = time.time()

            if timeslide_num * roll_amount > sample_length * SAMPLE_RATE:
                #going into degeneracy now, looped all the way around
                break
            # roll the segments - note that the windowing already happened
            #print("185", timeslide[:, :, 1, :].shape, RNN_precomputed['bbh'][:, 128:].shape)
            timeslide[:, :, 1, :] = torch.roll(timeslide[:, :, 1, :], roll_amount, dims=1)

            # now roll the intermediate LSTM values
            # 128 comes from the fact that they are stacked. x = torch.cat([Hx, Lx], dim=1), 
            # so Lx should have the latter indicies
            for key in ["bbh", "sghf", "sglf"]:
                RNN_precomputed[key][:, 128:] = torch.roll(RNN_precomputed[key][:, 128:], roll_amount, dims=0)
            #print("in evaluate timeslides, RNN computed value", RNN_precomputed['bbh'][0, 120:136])
            final_values, midpoints = full_evaluation(
                    timeslide, args.model_path, DEVICE, 
                    return_midpoints=True, loaded_models=gwak_models, grad_flag=False,
                    precomputed_rnn=RNN_precomputed, batch_size=batch_size_, already_split=True)

            # sanity check 
            sanity_check = False
            if sanity_check:
                final_values_, _ = full_evaluation(
                    segments_normalized, args.model_path, DEVICE, 
                    return_midpoints=True, loaded_models=gwak_models_, grad_flag=False, already_split=True)
                print("sanity check:", torch.mean(torch.abs(final_values - final_values_)))

            # remove the dummy batch dimension of 1
            final_values = final_values[0]

            save_full_timeslide_readout = True
            if save_full_timeslide_readout:

                FAR_2days = -1.67 # lowest FAR bin we want to worry about

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
               
                if len(indices) == 0: continue # just start the next timeslide, no interesting events
                
                indices = event_clustering(indices, smoothed_scores, 5*SAMPLE_RATE/SEGMENT_OVERLAP, DEVICE) # 5 seconds
                filtered_final_score = smoothed_scores.index_select(0, indices)
                filtered_final_scaled_evals = scaled_evals.index_select(0, indices)

                indices_ = indices.detach().cpu().numpy()

                # extract important timeslides with indices
                timeslide_chunks, edge_check_filter = extract_chunks(strain_data, timeslide_num, 
                                                    midpoints[indices_],
                                                    DEVICE, window_size=1024) # 0.25 seconds on either side
                                                                              # so it should come out to desired 0.5
                filtered_final_scaled_evals = filtered_final_scaled_evals.detach().cpu().numpy()
                filtered_final_score = filtered_final_score.detach().cpu().numpy()

                filtered_final_scaled_evals = filtered_final_scaled_evals[edge_check_filter]
                filtered_final_score = filtered_final_score[edge_check_filter]
                timeslide_chunks = timeslide_chunks[edge_check_filter]

                #print("386", filtered_final_scaled_evals.shape, timeslide_chunks.shape)
                if heuristics_tests:
                    combined_freqcorr = combine_freqcorr(filtered_final_scaled_evals)
                    passed_heuristics = []
                    for i, strain_segment in enumerate(timeslide_chunks):
                        passed_heuristics.append(joint_heuristic_test(strain_segment, combined_freqcorr[i],
                                                                      short_relation, long_relation))
                        
                    #print("passed heuristics:", passed_heuristics)

                    filtered_final_scaled_evals = filtered_final_scaled_evals[passed_heuristics]
                    filtered_final_score = filtered_final_score[passed_heuristics]
                    timeslide_chunks = timeslide_chunks[passed_heuristics]
                #print(important_timeslide.shape)
                #print(filtered_final_scaled_evals.shape, filtered_final_score.shape, timeslide_chunks.shape)
                #assert 0


                
                if len(indices_) > 0:
                    #print(len(indices_))
                    np.savez(f'{args.save_evals_path}/timeslide_evals_full_{timeslide_num}.npz',
                                                final_scaled_evals=filtered_final_scaled_evals,
                                                metric_score = filtered_final_score,
                                                timeslide_data = timeslide_chunks,
                                                time_event_ocurred = midpoints[indices_])

                #np.save(f'{args.save_evals_path}/sanity_timeslide_data.npy', strain_data)
                #print("event score", filtered_final_score)
                #print("gwak stats", filtered_final_scaled_evals)
                #print("saving folder", args.save_evals_path)
                
                #assert 0
                    
            if not os.path.exists(f"{args.save_evals_path}/livetime_tracker.npy"):
                track_data = np.zeros(1)
                np.save(f"{args.save_evals_path}/livetime_tracker.npy", track_data)
            tracked_sofar = np.load(f"{args.save_evals_path}/livetime_tracker.npy")
            np.save(f"{args.save_evals_path}/livetime_tracker.npy", tracked_sofar + reduced_len/SAMPLE_RATE)

            # save as a numpy file, with the index of timeslide_num
            np.save(f'{args.save_evals_path}/timeslide_evals_{timeslide_num}.npy', final_values.detach().cpu().numpy())

        # got out of the for loop, check why
        # either (likely) the for loop completed, and the desired number of timeslides was done
        # or getting degeneracy in the data, so now slide all of the data over by (1-200)
        if timeslide_num >= n_timeslides:
            not_finished = False
        else:
            n_timeslides -= timeslide_num
            initial_roll += 10 #in units of datapoints, something that incrementally gets to 200.
                                # could do 1, but then the inputs aren't so different, and with 10 this gives
                                # a maximum potential of ~172 years per hour, which should be enough
            # if initial_roll > 200...reduce it
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('model_path', nargs='+', type=str,
                        help='Path to the models')

    parser.add_argument('from_saved_models', type=bool,
                        help='If true, use the pre-trained models from MODELS_LOCATION in config, otherwise use models trained with the pipeline.')

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
