import os
import argparse
import numpy as np
import torch
import time
from typing import Optional
from quak_predict import quak_eval_snapshotter, quak_eval
from helper_functions import (
    std_normalizer_torch,
    split_into_segments_torch,
    stack_dict_into_tensor,
    pearson_computation
)

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append('/n/home00/emoreno/gw-anomaly/ml4gw')
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.utils.slicing import unfold_windows

from config import (
    SEGMENT_OVERLAP,
    GPU_NAME,
    CLASS_ORDER,
    DATA_EVAL_MAX_BATCH,
    SEG_NUM_TIMESTEPS,
    RETURN_INDIV_LOSSES,
    SCALE,
    SAMPLE_RATE,
    BATCH_SIZE,
    SEG_NUM_TIMESTEPS,
    BANDPASS_LOW, 
    NUM_IFOS
)

PSD_LENGTH = 64
FDURATION = 2
BATCH_SIZE = 2048
WINDOW_LENGTH = SEG_NUM_TIMESTEPS / SAMPLE_RATE
STRIDE = (SEG_NUM_TIMESTEPS - SEGMENT_OVERLAP) / SAMPLE_RATE

class BatchGenerator(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        sample_rate: float,
        kernel_length: float,
        fftlength: float,
        fduration: float,
        psd_length: float,
        inference_sampling_rate: float,
        highpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.spectral_density = SpectralDensity(
            sample_rate=SAMPLE_RATE,
            fftlength=fftlength,
            overlap=None,  # defaults to fftlength / 2
            average="median",
            fast=True  # not accurate for lowest 2 frequency bins, but we don't care about those
        )
        self.whitener = Whiten(
            fduration=fduration,
            sample_rate=sample_rate,
            highpass=highpass
        )

        self.step_size = int(sample_rate / inference_sampling_rate)
        self.kernel_size = int(kernel_length * sample_rate)
        self.fsize = int(fduration * sample_rate)
        self.psd_size = int(psd_length * sample_rate)
        self.num_channels = num_channels

    @property
    def state_size(self):
        return self.psd_size + self.kernel_size + self.fsize - self.step_size

    def get_initial_state(self):
        return torch.zeros((self.num_channels, self.state_size))

    def forward(self, X, state):
        state = torch.cat([state, X], dim=-1)
        split = [self.psd_size, state.size(-1) - self.psd_size]
        whiten_background, X = torch.split(state, split, dim=-1)

        # only use the PSD of the non-injected data for computing
        # our whitening to avoid biasing our PSD estimate
        psd = self.spectral_density(whiten_background.double())
        X = self.whitener(X, psd)
        X = unfold_windows(X, self.kernel_size, self.step_size)
        X = X.reshape(-1, self.num_channels, self.kernel_size)

        # divide by standard deviation along time axis
        X = X / X.std(axis=-1, keepdims=True)
        return X, state[:, -self.state_size :]

def full_evaluation_snapshotter(data, loaded_models, device, shift, return_midpoints=False):
    '''
    Passed in data is of shape (N_samples, 2, time_axis)
    '''

    batcher = BatchGenerator(
        num_channels=2,
        sample_rate=SAMPLE_RATE,
        kernel_length=WINDOW_LENGTH,
        fftlength=2,
        fduration=FDURATION,
        psd_length=PSD_LENGTH,
        inference_sampling_rate=1 / STRIDE,
        highpass=BANDPASS_LOW
    )
    batcher = batcher.to(device)

    quak_predictions_dict = quak_eval_snapshotter(
        data, loaded_models, batcher, device, shift)

    quak_predictions = stack_dict_into_tensor(  
        quak_predictions_dict, device=device)
    
    return quak_predictions


def full_evaluation(data, model_folder_path, device, return_midpoints=False, loaded_models=None, selection=None):
    '''
    Passed in data is of shape (N_samples, 2, time_axis)
    '''
    if not torch.is_tensor(data):
        data = torch.from_numpy(data).to(device)

    assert data.shape[1] == 2

    clipped_time_axis = (data.shape[2] // SEGMENT_OVERLAP) * SEGMENT_OVERLAP
    data = data[:, :, :clipped_time_axis]

    segments = split_into_segments_torch(data, device=device)
    
    slice_midpoints = np.arange(SEG_NUM_TIMESTEPS // 2, segments.shape[1] * (
        SEGMENT_OVERLAP) + SEG_NUM_TIMESTEPS // 2, SEGMENT_OVERLAP)

    if selection is not None:
        segments = segments[:, selection]
        slice_midpoints = slice_midpoints[selection.cpu().numpy()]

    segments_normalized = std_normalizer_torch(segments)

    # segments_normalized at this point is (N_batches, N_samples, 2, 100) and
    # must be reshaped into (N_batches * N_samples, 2, 100) to work with
    # quak_predictions
    N_batches, N_samples = segments_normalized.shape[
        0], segments_normalized.shape[1]
    segments_normalized = torch.reshape(
        segments_normalized, (N_batches * N_samples, 2, SEG_NUM_TIMESTEPS))
    
    quak_predictions_dict = quak_eval(
        segments_normalized, model_folder_path, device, loaded_models=loaded_models)

    quak_predictions = stack_dict_into_tensor(  
        quak_predictions_dict, device=device)

    if RETURN_INDIV_LOSSES:
        quak_predictions = torch.reshape(
            quak_predictions, (N_batches, N_samples, SCALE * len(CLASS_ORDER)))
    else:
        quak_predictions = torch.reshape(
            quak_predictions, (N_batches, N_samples, len(CLASS_ORDER)))
    
    # pearson 
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

    if return_midpoints:
        return final_values, slice_midpoints
    return final_values


def main(args):
    DEVICE = torch.device(GPU_NAME)
    data = np.load(args.data_path)['data']
    print(f'loaded data shape: {data.shape}')
    if data.shape[0] == 2:
        data = data.swapaxes(0, 1)
    n_batches_total = data.shape[0]

    _, timeaxis_size, feature_size = full_evaluation(
        data[:2], args.model_paths, DEVICE).cpu().numpy().shape
    result = np.zeros((n_batches_total, timeaxis_size, feature_size))
    n_splits = n_batches_total // DATA_EVAL_MAX_BATCH
    if n_splits * DATA_EVAL_MAX_BATCH != n_batches_total:
        n_splits += 1
    for i in range(n_splits):
        result[DATA_EVAL_MAX_BATCH * i:DATA_EVAL_MAX_BATCH * (i + 1)] = full_evaluation(
            data[DATA_EVAL_MAX_BATCH * i:DATA_EVAL_MAX_BATCH * (i + 1)], args.model_paths, DEVICE).cpu().numpy()

    np.save(args.save_path, result)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data_path', type=str,
                        help='Directory containing the injections to evaluate')

    parser.add_argument('save_path', type=str,
                        help='Folder to which save the evaluated injections')

    parser.add_argument('model_paths', nargs='+', type=str,
                        help='List of models')

    args = parser.parse_args()
    main(args)
