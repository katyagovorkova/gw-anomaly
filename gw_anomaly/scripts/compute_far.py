import os
import argparse
import numpy as np

import torch

from models import LinearModel
from evaluate_data import full_evaluation
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    FM_TIMESLIDE_TOTAL_DURATION,
    TIMESLIDE_TOTAL_DURATION,
    SAMPLE_RATE,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN,
    RETURN_INDIV_LOSSES,
    FACTORS_NOT_USED_FOR_FM,
    SMOOTHING_KERNEL_SIZES,
    DO_SMOOTHING,
    GPU_NAME,
    MODELS_LOCATION,
    FM_LOCATION
)


def main(args):

    #DEVICE = torch.device(f'cuda:{args.gpu}')
    DEVICE = torch.device(GPU_NAME)

    model_path = args.model_path if not args.from_saved_models else \
        [os.path.join(MODELS_LOCATION, os.path.basename(f)) for f in args.model_path]

    fm_model_path = args.fm_model_path if not args.from_saved_fm_model else \
        os.path.join(FM_LOCATION, os.path.basename(args.fm_model_path))

    metric_coefs_path = args.metric_coefs_path if not args.from_saved_fm_model else \
        os.path.join(FM_LOCATION, os.path.basename(args.metric_coefs_path))

    norm_factor_path = args.norm_factor_path if not args.from_saved_fm_model else \
        os.path.join(FM_LOCATION, os.path.basename(args.norm_factor_path))

    if metric_coefs_path is not None:
        # initialize histogram
        n_bins = 2 * int(HISTOGRAM_BIN_MIN / HISTOGRAM_BIN_DIVISION)

        if DO_SMOOTHING:
            for kernel_len in SMOOTHING_KERNEL_SIZES:
                mod_path = f'{args.save_path[:-4]}_{kernel_len}.npy'
                hist = np.zeros(n_bins)
                np.save(mod_path, hist)

        else:
            hist = np.zeros(n_bins)
            np.save(args.save_path, hist)


        # compute the dot product and save that instead
        metric_vals = np.load(metric_coefs_path)
        norm_factors = np.load(norm_factor_path)
        norm_factors_cpu = norm_factors[:] #copy
        metric_vals = torch.from_numpy(metric_vals).float().to(DEVICE)
        norm_factors = torch.from_numpy(norm_factors).float().to(DEVICE)

        model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)-1).to(DEVICE)
        model.load_state_dict(torch.load(
            fm_model_path, map_location=f'cuda:{args.gpu}'))

        learned_weights = model.layer.weight.detach().cpu().numpy()
        learned_bias = model.layer.bias.detach().cpu().numpy()

        def update_hist(vals):
            vals = np.array(vals)
            # a trick to not to re-evaluate saved timeslides
            vals = np.delete(vals, FACTORS_NOT_USED_FOR_FM, -1)
            vals = torch.from_numpy(vals).to(DEVICE)
            # flatten batch dimension
            vals = torch.reshape(vals, (vals.shape[
                                         0] * vals.shape[1], vals.shape[2]))
            means, stds = norm_factors[0], norm_factors[1]
            vals = (vals - means) / stds

            if RETURN_INDIV_LOSSES:
                model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
                model.load_state_dict(torch.load(
                    fm_model_path, map_location=f'cuda:{args.gpu}'))
                vals = model(vals).detach()
            else:
                vals = torch.matmul(vals, metric_vals)

            update = torch.histc(vals, bins=n_bins,
                                 min=-HISTOGRAM_BIN_MIN, max=HISTOGRAM_BIN_MIN)
            past_hist = np.load(args.save_path)
            new_hist = past_hist + update.cpu().numpy()
            np.save(args.save_path, new_hist)

        def update_hist_cpu(vals):
            vals = np.array(vals)
            # a trick to not to re-evaluate saved timeslides
            # vals = np.delete(vals, FACTORS_NOT_USED_FOR_FM, -1)
            #vals = torch.from_numpy(vals).to(DEVICE)
            # flatten batch dimension
            vals = np.reshape(vals, (vals.shape[
                                         0] * vals.shape[1], vals.shape[2]))
            means, stds = norm_factors_cpu[0], norm_factors_cpu[1]
            vals = (vals - means) / stds

            vals = np.matmul(vals, learned_weights.T) + learned_bias

            if DO_SMOOTHING:
                for kernel_len in SMOOTHING_KERNEL_SIZES:
                    if kernel_len == 1:
                        vals_convolved = vals
                    else:
                        kernel = np.ones((kernel_len)) / kernel_len
                        vals_convolved = np.convolve(vals[:, 0], kernel, mode='valid')

                    update,_ = np.histogram(vals_convolved, bins=n_bins, range=[-HISTOGRAM_BIN_MIN, HISTOGRAM_BIN_MIN])

                    mod_path = f"{args.save_path[:-4]}_{kernel_len}.npy"
                    past_hist = np.load(mod_path)
                    new_hist = past_hist + update
                    np.save(mod_path, new_hist)

            else:
                update,_ = np.histogram(vals, bins=n_bins, range=[-HISTOGRAM_BIN_MIN, HISTOGRAM_BIN_MIN])
                past_hist = np.load(args.save_path)
                new_hist = past_hist + update
                np.save(args.save_path, new_hist)

        # load pre-computed timeslides evaluations
        for folder in args.data_path:

            all_files = os.listdir(folder)
            print(f'Analyzing {folder} from {args.data_path}')

            for file_id in range(0,len(all_files)-len(all_files) % 5,5):

                if file_id%10000==0: print(f'Analyzing {file_id} from {len(all_files)}')
                all_vals = [ np.load(os.path.join(folder, all_files[file_id+local_id]))
                            for local_id in range(5)
                            if '.npy' in all_files[file_id+local_id] ]

                all_vals = np.concatenate(all_vals, axis=0)
                #update_hist(all_vals)
                update_hist_cpu(all_vals)

    else:
        print(150, DEVICE)
        data = np.load(args.data_path[0])['data']
        data = torch.from_numpy(data).to(DEVICE)

        reduction = 20  # for things to fit into memory nicely

        timeslide_total_duration = TIMESLIDE_TOTAL_DURATION
        if args.fm_shortened_timeslides:
            timeslide_total_duration = FM_TIMESLIDE_TOTAL_DURATION

        sample_length = data.shape[1] / SAMPLE_RATE
        n_timeslides = int(timeslide_total_duration //
                           sample_length) * reduction
        print('Number of timeslides:', n_timeslides)


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

        for timeslide_num in range(1, n_timeslides + 1):
            print(f'starting timeslide: {timeslide_num}/{n_timeslides}')

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
            final_values, _ = full_evaluation(
                timeslide[None, :, :], model_path, DEVICE)
            print(final_values.shape)
            print(213, final_values)
            print('saving, individually')
            means, stds = torch.mean(
                final_values, axis=-2), torch.std(final_values, axis=-2)
            means, stds = means.detach().cpu().numpy(), stds.detach().cpu().numpy()
            np.save(f'{args.save_normalizations_path}/normalization_params_{timeslide_num}.npy', np.stack([means, stds], axis=0))
            final_values = final_values.detach().cpu().numpy()
            if False:
                FAR_2days = -1.617 #lowest FAR bin we have
                norm_factors = np.load(f"/home/katya.govorkova/gwak-paper-final-models/trained/norm_factor_params.npy")
                fm_model_path = (f"/home/katya.govorkova/gwak-paper-final-models/trained/fm_model.pt")
                fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
                fm_model.load_state_dict(torch.load(
                    fm_model_path, map_location=GPU_NAME))#map_location=f'cuda:{args.gpu}'))
                linear_weights = fm_model.layer.weight.detach().cpu().numpy()
                bias_value = fm_model.layer.bias.detach().cpu().numpy()

                # Inferenc to save scores (final metric) and scaled_evals (GWAK space * weights unsummed)
                scores = []
                scaled_evals = []
                for elem in final_values[0]:
                    # elem = np.delete(elem, FACTORS_NOT_USED_FOR_FM, -1)
                    elem = (elem - norm_factors[0]) / norm_factors[1]
                    scaled_eval = np.multiply(elem, linear_weights)
                    #assert 0
                    scaled_evals.append(scaled_eval[0, :])
                    elem = torch.from_numpy(elem).to(DEVICE)
                    scores.append(fm_model(elem).detach().cpu().numpy())# - bias_value)

                scores = np.array(scores)
                scaled_evals = np.array(scaled_evals)

                indices = np.where(scores < FAR_2days)[0]
                filtered_final_score = scores[indices]
                filtered_final_scaled_evals = scaled_evals[indices]
                timeslide = timeslide.detach().cpu().numpy()

                #extract important timeslides with indices
                important_timeslide = extract_chunks(timeslide, indices, window_size=1024)

                #print(indices, filtered_final_score)
                #print("timeslide shape", important_timeslide.shape)
                #print("filtered_final_score", filtered_final_score.shape)
                #print("scaled_evals", filtered_final_scaled_evals.shape)

                np.savez(f'{args.save_evals_path}/timeslide_evals_FULL_{timeslide_num}.npz',
                                             final_scaled_evals=filtered_final_scaled_evals,
                                             metric_score = filtered_final_score,
                                             timeslide_data = important_timeslide)

            # save as a numpy file, with the index of timeslide_num
            
            np.save(f'{args.save_evals_path}/timeslide_evals_{timeslide_num}.npy', final_values)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('save_path', type=str,
                        help='Folder to which save the timeslides')

    parser.add_argument('model_path', nargs='+', type=str,
                        help='Path to the models')

    parser.add_argument('from_saved_models', type=bool,
                        help='If true, use the pre-trained models from MODELS_LOCATION in config, otherwise use models trained with the pipeline.')

    # Additional arguments
    parser.add_argument('--data-path', type=str, nargs='+',
                        help='Directory containing the timeslides')

    parser.add_argument('--fm-model-path', type=str,
                        help='Final metric model')

    parser.add_argument('--from-saved-fm-model', type=bool,
                        help='If true, use the pre-trained models from MODELS_LOCATION in config, otherwise use models trained with the pipeline.')

    parser.add_argument('--metric-coefs-path', type=str, default=None,
                        help='Pass in path to metric coefficients to compute dot product')

    parser.add_argument('--norm-factor-path', type=str, default=None,
                        help='Pass in path to significance normalization factors')

    parser.add_argument('--fm-shortened-timeslides', type=str, default='False',
                        help='Generate reduced timeslide samples to train final metric')

    parser.add_argument('--gpu', type=str, default='1',
                        help='On which GPU to run')

    parser.add_argument('--save-evals-path', type=str, default=None,
                        help='Where to save evals')

    parser.add_argument('--save-normalizations-path', type=str, default=None,
                        help='Where to save normalizations')

    args = parser.parse_args()
    args.fm_shortened_timeslides = args.fm_shortened_timeslides == 'True'

    main(args)
