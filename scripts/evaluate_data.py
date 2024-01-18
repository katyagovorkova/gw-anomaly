import os
import argparse
import numpy as np
import torch

from qwak_predict import qwak_eval
from helper_functions import (
    std_normalizer_torch,
    split_into_segments_torch,
    stack_dict_into_tensor,
    pearson_computation
)
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SEGMENT_OVERLAP,
    GPU_NAME,
    CLASS_ORDER,
    DATA_EVAL_MAX_BATCH,
    SEG_NUM_TIMESTEPS,
    RETURN_INDIV_LOSSES,
    SCALE,
    MODELS_LOCATION
)


def full_evaluation(data, model_folder_path, device, return_midpoints=False,
                    loaded_models=None, selection=None, grad_flag=True,
                    already_split=False, precomputed_rnn=None, batch_size=None, do_rnn_precomp=False):
    '''
    Passed in data is of shape (N_samples, 2, time_axis)
    '''
    #t33 = time.time()

    if not already_split:
        if not torch.is_tensor(data):
            data = torch.from_numpy(data).to(device)
        assert data.shape[1] == 2
        clipped_time_axis = (data.shape[2] // SEGMENT_OVERLAP) * SEGMENT_OVERLAP
        data = data[:, :, :clipped_time_axis]

        segments = split_into_segments_torch(data, device=device)
        #print("Slicing data time, ", time.time()-t33)
        segments_normalized = std_normalizer_torch(segments)

    else:
        segments_normalized = data

    slice_midpoints = np.arange(SEG_NUM_TIMESTEPS // 2, segments_normalized.shape[1] * (  #if something fails, segments normalized has
        SEGMENT_OVERLAP) + SEG_NUM_TIMESTEPS // 2, SEGMENT_OVERLAP)                         # a different shape

    if selection is not None:
        segments = segments[:, selection]
        slice_midpoints = slice_midpoints[selection.cpu().numpy()]

    # segments_normalized at this point is (N_batches, N_samples, 2, 100) and
    # must be reshaped into (N_batches * N_samples, 2, 100) to work with
    # quak_predictions
    N_batches, N_samples = segments_normalized.shape[
        0], segments_normalized.shape[1]
    segments_normalized = torch.reshape(
        segments_normalized, (N_batches * N_samples, 2, SEG_NUM_TIMESTEPS))

    #print("59 evaluate data grad_flag:", grad_flag)
    #t61 = time.time()
    quak_predictions_dict = quak_eval(
        segments_normalized, model_folder_path, device, loaded_models=loaded_models,
        grad_flag = grad_flag, precomputed_rnn=precomputed_rnn, batch_size=batch_size, do_rnn_precomp=do_rnn_precomp)
    if do_rnn_precomp:
        return quak_predictions_dict

    #print("quak eval time", time.time()-t61, quak_predictions_dict[list(quak_predictions_dict.keys())[0]][0][0])
    #t65 = time.time()

    quak_predictions = stack_dict_into_tensor(
        quak_predictions_dict, device=device)
    #print("stacking time,", time.time()-t65)

    if RETURN_INDIV_LOSSES:
        quak_predictions = torch.reshape(
            quak_predictions, (N_batches, N_samples, SCALE * len(CLASS_ORDER)))
    else:
        quak_predictions = torch.reshape(
            quak_predictions, (N_batches, N_samples, len(CLASS_ORDER)))

    if PEARSON_FLAG:
        pearson_values, (edge_start, edge_end) = pearson_computation(data, device)

        # may need to manually override this
        pearson_values = pearson_values[:, :, None]
        if edge_end-edge_start > 0:
            disparity = quak_predictions.shape[1] - pearson_values.shape[1]
            edge_start = disparity//2
            edge_end = -disparity//2

        if edge_end != 0:
            quak_predictions = quak_predictions[:, edge_start:edge_end, :]
            slice_midpoints = slice_midpoints[edge_start:edge_end]
        else:
            quak_predictions = quak_predictions[:, edge_start:, :]
            slice_midpoints = slice_midpoints[edge_start:]
        if selection != None:
            final_values = torch.cat([quak_predictions, pearson_values[:, selection]], dim=-1)
        else:
            final_values = torch.cat([quak_predictions, pearson_values], dim=-1)

    return final_values, slice_midpoints


def main(args):

    DEVICE = torch.device(GPU_NAME)

    model_path = args.model_path if not args.from_saved_models else \
        [os.path.join(MODELS_LOCATION, os.path.basename(f)) for f in args.model_path]

    data = np.load(args.data_path)['data']
    print(f'loaded data shape: {data.shape}')
    if data.shape[0] == 2:
        data = data.swapaxes(0, 1)
    n_batches_total = data.shape[0]

    _, timeaxis_size, feature_size, _ = full_evaluation(
        data[:2], model_path, DEVICE).cpu().numpy().shape
    result = np.zeros((n_batches_total, timeaxis_size, feature_size))
    n_splits = n_batches_total // DATA_EVAL_MAX_BATCH
    if n_splits * DATA_EVAL_MAX_BATCH != n_batches_total:
        n_splits += 1
    for i in range(n_splits):
        result[DATA_EVAL_MAX_BATCH * i:DATA_EVAL_MAX_BATCH * (i + 1)] = full_evaluation(
            data[DATA_EVAL_MAX_BATCH * i:DATA_EVAL_MAX_BATCH * (i + 1)], model_path, DEVICE).cpu().numpy()

    np.save(args.save_path, result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', type=str,
                        help='Directory containing the injections to evaluate')

    parser.add_argument('save_path', type=str,
                        help='Folder to which save the evaluated injections')

    parser.add_argument('model_path', nargs='+', type=str,
                        help='List of models')

    parser.add_argument('from_saved_models', type=bool,
                        help='If true, use the pre-trained models from MODELS_LOCATION in config, otherwise use models trained with the pipeline.')

    args = parser.parse_args()
    main(args)
