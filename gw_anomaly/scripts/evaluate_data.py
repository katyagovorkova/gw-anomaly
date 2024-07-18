import os
import argparse
import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import pearsonr
from gw_anomaly.scripts.models import LinearModel
from gw_anomaly.scripts.gwak_predict import quak_eval
from gw_anomaly.scripts.helper_functions import (
    std_normalizer_torch,
    split_into_segments_torch,
    stack_dict_into_tensor,
    pearson_computation,
    joint_heuristic_test,
    combine_freqcorr
)
# import sys
# sys.path.append(
#     os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    VERSION,
    SEGMENT_OVERLAP,
    GPU_NAME,
    CLASS_ORDER,
    DATA_EVAL_MAX_BATCH,
    SEG_NUM_TIMESTEPS,
    RETURN_INDIV_LOSSES,
    SCALE,
    MODELS_LOCATION,
    PEARSON_FLAG,
    DATA_EVAL_USE_HEURISTIC,
    FACTORS_NOT_USED_FOR_FM,

)
def compute_signal_strength_chop_sep(x, y):
    psd0 = welch(x)[1]
    psd1 = welch(y)[1]
    HLS = np.log(np.sum(psd0))
    LLS = np.log(np.sum(psd1))
    return HLS, LLS
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

import torch.nn as nn
class BasedModel(nn.Module):
    def __init__(self):
        super(BasedModel, self).__init__()

        self.layer1 = nn.Linear(3, 1)
        self.layer2_1 = nn.Linear(1, 1)
        self.layer2_2 = nn.Linear(1, 1)
        self.layer2_3 = nn.Linear(1, 1)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.activation(self.layer1(x[:, :3]))
        x2_1 = self.activation(self.layer2_1(x[:, 3:4]))
        x2_2 = self.activation(self.layer2_1(x[:, 4:5]))
        x2_3 = self.activation(self.layer2_1(x[:, 5:6]))
        return x1 * x2_1 * x2_2 * x2_3

def extract(gwak_values):
    print(74, "in", gwak_values.shape)
    result = np.zeros((gwak_values.shape[0], 3))
    for i, pair in enumerate([[3, 4], [9, 10], [12, 13]]):
        a, b = pair
        ratio_a = (np.abs(gwak_values[:, a]) + 2) / (np.abs(gwak_values[:, b]) + 2)
        ratio_b = (np.abs(gwak_values[:, b]) + 2) / (np.abs(gwak_values[:, a]) + 2)

        ratio = np.maximum(ratio_a, ratio_b)
        print(81, ratio.shape)
        result[:, i] = ratio

    return result


def parse_strain(x):
    # take strain, compute the long sig strenght & pearson
    # split it up, do the same thing for short
    long_pearson, shift_idx = shifted_pearson(x[0], x[1], 50, len(x[0])-50)
    #long_sig_strength = compute_signal_strength_chop(x[0, 50:-50], x[1, 50+shift_idx:len(x[0])-50+shift_idx] )
    HSS, LSS = compute_signal_strength_chop_sep(x[0, 50:-50], x[1, 50+shift_idx:len(x[0])-50+shift_idx])
    return long_pearson, HSS, LSS

def full_evaluation(data_, model_folder_path, device, return_midpoints=False,
                    loaded_models=None, selection=None, grad_flag=True,
                    already_split=False, precomputed_rnn=None, batch_size=None,
                    do_rnn_precomp=False, return_recreations=False, do_normalization=False):
    '''
    Passed in data is of shape (N_samples, 2, time_axis)
    '''
    #t33 = time.time()

    if not already_split:
        if not torch.is_tensor(data_):
            data_ = torch.from_numpy(data_).to(device)
        assert data_.shape[1] == 2
        print(110, data_.shape)
        clipped_time_axis = (data_.shape[2] // SEGMENT_OVERLAP) * SEGMENT_OVERLAP
        data = data_[:][:, :, :clipped_time_axis]
        print(113, data.shape)
        segments = split_into_segments_torch(data, device=device)
        #print("Slicing data time, ", time.time()-t33)
        segments_normalized = std_normalizer_torch(segments)

    else:
        if do_normalization:
            data = std_normalizer_torch(data_)
        segments_normalized = data_

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
    print(134, segments_normalized.shape)
    segments_normalized = torch.reshape(
        segments_normalized, (N_batches * N_samples, 2, SEG_NUM_TIMESTEPS))

    #print("59 evaluate data grad_flag:", grad_flag)
    #t61 = time.time()
    quak_predictions_dict = quak_eval(
        segments_normalized, model_folder_path, device, loaded_models=loaded_models,
        grad_flag = grad_flag, precomputed_rnn=precomputed_rnn, batch_size=batch_size,
        do_rnn_precomp=do_rnn_precomp, reduce_loss= not(return_recreations) )
    if do_rnn_precomp:
        return quak_predictions_dict

    #print("quak eval time", time.time()-t61, quak_predictions_dict[list(quak_predictions_dict.keys())[0]][0][0])
    #t65 = time.time()
    if return_recreations:
        quak_predictions = stack_dict_into_tensor(
            quak_predictions_dict['freq_loss'], device=device)
        originals = quak_predictions_dict['original']
        recreations = quak_predictions_dict['recreated']

    else:
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
    if return_recreations:
        return quak_predictions, slice_midpoints, originals, recreations
    return quak_predictions, slice_midpoints


def main(args):

    DEVICE = torch.device(GPU_NAME)

    model_path = args.model_path if not args.from_saved_models else \
        [os.path.join(MODELS_LOCATION, os.path.basename(f)) for f in args.model_path]
    #print(args.data_path)
    #assert 0
    
    data = np.load(args.data_path)['data']
    print("data path", args.data_path)
    print(f'loaded data shape: {data.shape}')
    if data.shape[0] == 5:
        # generated by curriculum?
        data = data[-1]
    if len(data.shape) == 2:
        data = data[None, :, :]
    if data.shape[0] == 2:
        data = data.swapaxes(0, 1)
    n_batches_total = data.shape[0]
    
    _, timeaxis_size, feature_size = full_evaluation(
        data[:5], model_path, DEVICE)[0].cpu().numpy().shape
    result = np.zeros((n_batches_total, timeaxis_size, feature_size))
    n_splits = n_batches_total // DATA_EVAL_MAX_BATCH
    if n_splits * DATA_EVAL_MAX_BATCH != n_batches_total:
        n_splits += 1
    for i in range(n_splits):
        output, midpoints = full_evaluation(
            data[DATA_EVAL_MAX_BATCH * i:DATA_EVAL_MAX_BATCH * (i + 1)], model_path, DEVICE)#[0].cpu().numpy()
        result[DATA_EVAL_MAX_BATCH * i:DATA_EVAL_MAX_BATCH * (i + 1)] = output.cpu().numpy()
    np.save(args.save_path, result)

    model_path = "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/plots/model.h5"
    model_heuristic = BasedModel().to(DEVICE)
    model_heuristic.load_state_dict(torch.load(model_path))

    data_eval_use_heuristic = False
    if data_eval_use_heuristic:
        SNRs = np.load(f"{args.data_path[:-4]}_SNR.npz.npy")
        # need to get the point of highest score

        norm_factors = np.load(f"output/{VERSION}/trained/norm_factor_params.npy")

        fm_model_path = (f"output/{VERSION}/trained/fm_model.pt")
        fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
        fm_model.load_state_dict(torch.load(
            fm_model_path, map_location=GPU_NAME))

        linear_weights = fm_model.layer.weight.detach()#.cpu().numpy()
        # linear_weights[:, -2] += linear_weights[:, -1]
        # linear_weights = linear_weights[:, :-1]
        # norm_factors = norm_factors[:, :-1]

        result = torch.from_numpy((result-norm_factors[0])/norm_factors[1]).float().to(DEVICE)
        scaled_evals = torch.multiply(result, linear_weights[None, :])#[0, :]
        scores = (scaled_evals.sum(axis=2))#[:, None]
        scores = scores.detach().cpu().numpy()
        strongest = np.argmin(scores, axis=1)
        scaled_evals = scaled_evals.detach().cpu().numpy()
        #long_relation = np.load("/home/ryan.raikman/share/gwak/long_relation.npy")
        #short_relation = np.load("/home/ryan.raikman/share/gwak/short_relation.npy")
        passed = []
        #build_dataset_strain = []
        build_heur_model_evals = []
        build_data_gwak_features = []
        SNRs__ = []
        print("201", scaled_evals.shape)
        #gwak_filtered = extract(scaled_evals)
        for i in range(len(data)):
            #strain_center = midpoints[strongest[i]]
            #print(midpoints[strongest[i]], strongest[i])
            strain_center = 7300
            eval_strongest_loc = 144
            if args.save_path[:-4].split("/")[-1] == "wnbhf_varying_snr_evals":
                strain_center = 9300 
                eval_strongest_loc = 184
            elif args.save_path[:-4].split("/")[-1] == "wnblf_varying_snr_evals":
                strain_center = 9400 
                eval_strongest_loc = 186
            elif args.save_path[:-4].split("/")[-1] == "supernova_varying_snr_evals":
                strain_center = 10850
                eval_strongest_loc = 218

            gwak_filtered = extract(scaled_evals[:, eval_strongest_loc, :])

            #if SNRs[i] > 12:
            SNRs__.append(SNRs[i])
            seg = data[i, :, strain_center-1024:strain_center+1024]
            strain_feats = parse_strain(seg)
            together = np.concatenate([strain_feats, gwak_filtered[i]])
            heur_res = model_heuristic(torch.from_numpy(together[None, :]).float().to(DEVICE)).item()
            build_heur_model_evals.append(heur_res)
            # gwak_filtered
            # build_dataset_strain.append([pearson_, HSS, LSS])
            build_data_gwak_features.append(scaled_evals[i][eval_strongest_loc])
            print(f"Computing heuristic test {i}/{len(data)}, SNR {SNRs[i]}" , end = '\r')

        # save the heuristic cut dataset
        heuristic_dir = f"{os.path.dirname(args.save_path)}/heuristic/"
        print("heuritic dir", heuristic_dir)
        try:
            os.makedirs(heuristic_dir)
        except FileExistsError:
            None
        class_name = args.save_path.split("/")[-1].split("_")[0]
        np.save(f"{heuristic_dir}/SIG_EVAL_{class_name}_heur_model_evals.npy", np.array(build_heur_model_evals))
        np.save(f"{heuristic_dir}/SIG_EVAL{class_name}_gwak_feats.npy", np.array(build_data_gwak_features))
        np.save(f"{heuristic_dir}/SIG_EVAL{class_name}_SNRs.npy", np.array(SNRs__))

        print("heuristic dir", heuristic_dir)
        #passed = np.array(passed)
        #print("177", passed.sum())

        #np.save(f"{args.save_path[:-4]}_heuristic_res.npy", passed)

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
