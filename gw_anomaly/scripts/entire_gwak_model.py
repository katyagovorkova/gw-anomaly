import os
import numpy as np
import argparse
import torch
from gwpy.timeseries import TimeSeries
from astropy import units as u
from scipy.signal import welch
import matplotlib.pyplot as plt
from scripts.models import LinearModel, GwakClassifier
from scripts.evaluate_data import full_evaluation
import json
import matplotlib
from torch.nn.functional import conv1d
from scipy.stats import pearsonr
import pickle
from scripts.helper_functions import far_to_metric, compute_fars, \
                            load_gwak_models, joint_heuristic_test, \
                                combine_freqcorr
from config import (
    FM_LOCATION,
    CHANNEL,
    GPU_NAME,
    SEGMENT_OVERLAP,
    SAMPLE_RATE,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    FACTORS_NOT_USED_FOR_FM,
    MODELS_LOCATION,
    SEG_NUM_TIMESTEPS,
    CLASS_ORDER,
    BOTTLENECK,
    MODEL,

    )
DEVICE = torch.device(GPU_NAME)
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
    result = np.zeros((gwak_values.shape[0], 3))
    for i, pair in enumerate([[3, 4], [9, 10], [12, 13]]):
        a, b = pair
        ratio_a = (np.abs(gwak_values[:, a]) + 2) / (np.abs(gwak_values[:, b]) + 2)
        ratio_b = (np.abs(gwak_values[:, b]) + 2) / (np.abs(gwak_values[:, a]) + 2)

        ratio = np.maximum(ratio_a, ratio_b)
        result[:, i] = ratio
    return result

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

def parse_strain(x):
    # take strain, compute the long sig strenght & pearson
    # split it up, do the same thing for short
    long_pearson, shift_idx = shifted_pearson(x[0], x[1], 50, len(x[0])-50)
    #long_sig_strength = compute_signal_strength_chop(x[0, 50:-50], x[1, 50+shift_idx:len(x[0])-50+shift_idx] )
    HSS, LSS = compute_signal_strength_chop_sep(x[0, 50:-50], x[1, 50+shift_idx:len(x[0])-50+shift_idx])
    return long_pearson, HSS, LSS

def sig_prob_function(evals, scale=40):
    sigmoid = lambda x: 1/(1+np.exp(-x))
    return 1-(sigmoid(scale * (evals-0.5)))

class FullGWAK():
    def __init__(self, models_location):
        self.load_models(models_location)
        self.load_auxiliary()
        #self.load_heuristic()


    def load_models(self, models_location):
        # load the GWAK autoencoders
        model_path = [f"{MODELS_LOCATION}/bbh.pt",
                      f"{MODELS_LOCATION}/sglf.pt",
                      f"{MODELS_LOCATION}/sghf.pt",
                      f"{MODELS_LOCATION}/background.pt",
                      f"{MODELS_LOCATION}/glitches.pt"]
        self.gwak_models = load_gwak_models(models_path, DEVICE, GPU_NAME)
        self.models_path = models_path

    def load_auxiliary(self):
        # initialize the smoothing kernel
        # load the final metric models
        # load the normalizing factors
        orig_kernel = 50
        kernel_len = int(orig_kernel * 5/SEGMENT_OVERLAP)
        kernel = torch.ones((1, kernel_len)).float().to(DEVICE)/kernel_len
        self.kernel = kernel[None, :, :]

        norm_factors = np.load(f"{FM_LOCATION}/norm_factor_params.npy")

        fm_model_path = (f"{FM_LOCATION}/fm_model.pt")
        fm_model = LinearModel(21-len(FACTORS_NOT_USED_FOR_FM)).to(DEVICE)
        fm_model.load_state_dict(torch.load(
            fm_model_path, map_location=GPU_NAME))

        linear_weights = fm_model.layer.weight.detach()#.cpu().numpy()
        self.bias_value = fm_model.layer.bias.detach()#.cpu().numpy()
        linear_weights[:, -2] += linear_weights[:, -1]
        self.linear_weights = linear_weights[:, :-1]
        norm_factors = norm_factors[:, :-1]
        self.mean_norm = torch.from_numpy(norm_factors[0]).to(DEVICE)#[:-1]
        self.std_norm = torch.from_numpy(norm_factors[1]).to(DEVICE)#[:-1]

    def load_heuristic(self):
        model_path = f"{MODELS_LOCATION}/model_heuristic.h5"
        self.model_heuristic = BasedModel().to(DEVICE)
        self.model_heuristic.load_state_dict(torch.load(model_path, map_location=DEVICE))

    def evaluate_data(self, data):

        # given data, run the full gwak evaluation
        final_values, midpoints, original, recreated = full_evaluation(  
                        data[None, :, :, :], self.models_path, DEVICE,
                        return_midpoints=True, return_recreations=True,
                        loaded_models=self.gwak_models, grad_flag=False,
                        already_split=True, do_normalization=True)
    
        final_values = final_values[0]

        # get the gwak scores
        final_values_slx = (final_values - self.mean_norm)/self.std_norm
        scaled_evals = torch.multiply(final_values_slx, self.linear_weights[None, :])[0, :]
        #print("scaled_evals", scaled_evals.shape)
        scores = (scaled_evals.sum(axis=1) + self.bias_value)[:, None]

        return scores, final_values
