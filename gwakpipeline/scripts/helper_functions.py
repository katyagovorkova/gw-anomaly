import os
import sys
import numpy as np
import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SAMPLE_RATE,
    SEG_NUM_TIMESTEPS,
    SEGMENT_OVERLAP,
    CLASS_ORDER,
    MAX_SHIFT,
    SHIFT_STEP,
    RETURN_INDIV_LOSSES,
    SCALE,
)


def mae_torch(a, b):
    # compute the MAE, do .mean twice:
    # once for the time axis, second time for detector axis
    return torch.abs(a - b).mean(axis=-1).mean(axis=-1)


def freq_loss_torch(a, b):
    a_ = torch.fft.rfft(a, axis=-1)
    b_ = torch.fft.rfft(b, axis=-1)
    a2b = torch.abs(torch.linalg.vecdot(a_, b_, axis=-1))
    a2a = torch.abs(torch.linalg.vecdot(
        a_[:, 0, :], a_[:, 1, :], axis=-1))[:, None]
    # b2b = torch.abs(torch.linalg.vecdot(
    #     b_[:, 0, :], b_[:, 1, :], axis=-1))[:, None]
    # return torch.hstack([a2b, a2a, b2b])
    return torch.hstack([a2b, a2a])


def std_normalizer_torch(data):
    std_vals = torch.std(data, dim=-1)[:, :, :, None]
    return data / std_vals


def stack_dict_into_tensor(data_dict, device=None):
    '''
    Input is a dictionary of keys, stack it into *torch* tensor
    '''
    fill_len = len(data_dict['bbh'])
    if RETURN_INDIV_LOSSES:
        stacked_tensor = torch.empty(
            (fill_len, len(CLASS_ORDER) * SCALE), device=device)
    else:
        stacked_tensor = torch.empty(
            (fill_len, len(CLASS_ORDER)), device=device)
    for class_name in data_dict.keys():
        stack_index = CLASS_ORDER.index(class_name)

        if RETURN_INDIV_LOSSES:
            stacked_tensor[:, stack_index * SCALE:stack_index *
                           SCALE + SCALE] = data_dict[class_name]
        else:
            stacked_tensor[:, stack_index] = data_dict[class_name]

    return stacked_tensor


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


def pearson_computation(data,
                        device,
                        max_shift=MAX_SHIFT,
                        seg_len=SEG_NUM_TIMESTEPS,
                        seg_step=SEGMENT_OVERLAP,
                        shift_step=SHIFT_STEP):
    max_shift = int(max_shift * SAMPLE_RATE)
    offset_families = np.arange(max_shift, max_shift + seg_len, seg_step)

    feature_length_full = data.shape[-1]
    feature_length = (data.shape[-1] // SEG_NUM_TIMESTEPS) * SEG_NUM_TIMESTEPS
    n_manual = (feature_length_full - feature_length) // seg_step
    n_batches = data.shape[0]
    data[:, 1, :] = -1 * data[:, 1, :]  # inverting livingston
    family_count = len(offset_families)
    final_length = 0
    for family_index in range(family_count):
        end = feature_length - seg_len + offset_families[family_index]
        if end > feature_length - max_shift:
            # correction: reduce by 1
            final_length -= 1
        final_length += (feature_length - seg_len) // seg_len
    family_fill_max = final_length
    final_length += n_manual
    all_corrs = torch.zeros((n_batches, final_length), device=device)
    for family_index in range(family_count):
        end = feature_length - seg_len + offset_families[family_index]
        if end > feature_length - max_shift:
            end -= seg_len
        hanford = data[:, 0, offset_families[family_index]:end].reshape(n_batches, -1, SEG_NUM_TIMESTEPS)
        hanford = hanford - hanford.mean(dim=2)[:, :, None]
        best_corrs = -1 * \
            torch.ones((hanford.shape[0], hanford.shape[1]), device=device)
        for shift_amount in np.arange(-max_shift, max_shift + shift_step, shift_step):

            livingston = data[:, 1, offset_families[
                family_index] + shift_amount:end + shift_amount].reshape(n_batches, -1, SEG_NUM_TIMESTEPS)
            livingston = livingston - livingston.mean(dim=2)[:, :, None]

            corrs = torch.sum(hanford * livingston, axis=2) / torch.sqrt(torch.sum(
                hanford * hanford, axis=2) * torch.sum(livingston * livingston, axis=2))
            best_corrs = torch.maximum(best_corrs, corrs)

        all_corrs[:, family_index:family_fill_max:family_count] = best_corrs

    # fill in pieces left over at end
    for k, center in enumerate(np.arange(feature_length - max_shift - seg_len // 2 + seg_step,
                                         feature_length_full - max_shift - seg_len // 2 + seg_step,
                                         seg_step)):
        hanford = data[:, 0, center - seg_len // 2:center + seg_len // 2]
        hanford = hanford - hanford.mean(dim=1)[:, None]
        best_corr = -1 * torch.ones((n_batches), device=device)
        for shift_amount in np.arange(-max_shift, max_shift + shift_step, shift_step):
            livingston = data[:, 1, center - seg_len // 2 +
                              shift_amount:center + seg_len // 2 + shift_amount]
            livingston = livingston - livingston.mean(dim=1)[:, None]
            corr = torch.sum(hanford * livingston, axis=1) / torch.sqrt(torch.sum(
                hanford * hanford, axis=1) * torch.sum(livingston * livingston, axis=1))
            best_corr = torch.maximum(best_corr, corr)
        all_corrs[:, -(k + 1)] = best_corr

    edge_start, edge_end = max_shift // seg_step, -(max_shift // seg_step) + 1
    return all_corrs, (edge_start, edge_end)
