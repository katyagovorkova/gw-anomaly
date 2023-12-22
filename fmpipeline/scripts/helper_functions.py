import os
import sys
import h5py
import time
import bilby
import numpy as np
import torch

from scipy.stats import cosine as cosine_distribution
from gwpy.timeseries import TimeSeries
from lalinference import BurstSineGaussian, BurstSineGaussianF

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from models import LSTM_AE_SPLIT, FAT, LinearModel

from config import (
    IFOS,
    SAMPLE_RATE,
    EDGE_INJECT_SPACING,
    GLITCH_SNR_BAR,
    STRAIN_START,
    STRAIN_STOP,
    LOADED_DATA_SAMPLE_RATE,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    GW_EVENT_CLEANING_WINDOW,
    SEG_NUM_TIMESTEPS,
    SEGMENT_OVERLAP,
    CLASS_ORDER,
    SIGNIFICANCE_NORMALIZATION_DURATION,
    GPU_NAME,
    MAX_SHIFT,
    SHIFT_STEP,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN,
    CHANNEL,
    NUM_IFOS,
    RETURN_INDIV_LOSSES,
    SCALE,
    MODEL,
    BOTTLENECK, 
    FACTORS_NOT_USED_FOR_FM,
    SMOOTHING_KERNEL)


def std_normalizer_torch(data):
    std_vals = torch.std(data, dim=-1)[:, :, :, None]
    return data / std_vals


def load_gwak_models(model_path, device, device_name):
    loaded_models = {}
    for dpath in model_path:
        model_name = dpath.split("/")[-1].split(".")[0]
        if MODEL[model_name] == "lstm":
            model = LSTM_AE_SPLIT(num_ifos=NUM_IFOS,
                                num_timesteps=SEG_NUM_TIMESTEPS,
                                BOTTLENECK=BOTTLENECK[model_name]).to(device)
        elif MODEL[model_name] == "dense":
            model = FAT(num_ifos=NUM_IFOS,
                        num_timesteps=SEG_NUM_TIMESTEPS,
                        BOTTLENECK=BOTTLENECK[model_name]).to(device)

        model.load_state_dict(torch.load(dpath, map_location=device_name))
        loaded_models[dpath] = model

    return loaded_models


def split_into_segments_torch(data,
                              overlap=SEGMENT_OVERLAP,
                              seg_len=SEG_NUM_TIMESTEPS,
                              device=None):
    '''
    Function to slice up data into overlapping segments
    seg_len: length of resulting segments
    overlap: overlap of the windows in units of indicies

    assuming that data is of shape (N_samples, 2, feature_len)
    '''
    N_slices = (data.shape[2] - seg_len) // overlap
    data = data[:, :, :N_slices * overlap + seg_len]
    feature_length_full = data.shape[2]
    feature_length = (data.shape[2] // SEG_NUM_TIMESTEPS) * SEG_NUM_TIMESTEPS
    N_slices_limited = (feature_length - seg_len) // overlap
    n_batches = data.shape[0]
    n_detectors = data.shape[1]

    # resulting shape: (batches, N_slices, 2, SEG_NUM_TIMESTEPS)
    result = torch.empty((n_batches, N_slices, n_detectors, seg_len),
                         device=device)

    offset_families = np.arange(0, seg_len, overlap)
    family_count = len(offset_families)
    final_length = 0
    for family_index in range(family_count):
        end = feature_length - seg_len + offset_families[family_index]
        if end > feature_length:
            # correction: reduce by 1
            final_length -= 1
        final_length += (feature_length - seg_len) // seg_len

    for family_index in range(family_count):
        end = feature_length - seg_len + offset_families[family_index]
        if end > feature_length:
            end -= seg_len
        result[:, family_index:N_slices_limited:family_count, 0, :] = data[
            :, 0, offset_families[family_index]:end].reshape(n_batches, -1, SEG_NUM_TIMESTEPS)
        result[:, family_index:N_slices_limited:family_count, 1, :] = data[
            :, 1, offset_families[family_index]:end].reshape(n_batches, -1, SEG_NUM_TIMESTEPS)
    # do the pieces left over, at the end
    for i in range(1, N_slices - N_slices_limited + 1):
        end = int(feature_length_full - i * overlap)
        start = int(end - seg_len)
        result[:, -i, :, :] = data[:, :, start:end]

    return result
