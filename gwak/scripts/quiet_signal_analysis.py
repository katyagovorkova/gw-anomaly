import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    GPU_NAME,
    SVM_LR,
    N_SVM_EPOCHS,
    RETURN_INDIV_LOSSES,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN,
    )
DEVICE = torch.device(GPU_NAME)


class LinearModel(nn.Module):
    def __init__(self, n_dims):
        super(LinearModel, self).__init__()
        self.layer_1 = nn.Linear(21, 32)
        self.layer_2 = nn.Linear(32, 32)
        self.layer_3 = nn.Linear(32, 1)
        
    def forward(self, x):
        
        x = F.tanh(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)
        

def optimize_hyperplane(signals, backgrounds):
    # saved as a batch, which needs to be flattned out
    backgrounds = np.reshape(backgrounds, (backgrounds.shape[0]*backgrounds.shape[1], backgrounds.shape[2]))

    sigs = torch.from_numpy(signals).float().to(DEVICE)
    bkgs = torch.from_numpy(backgrounds).float().to(DEVICE)
    network = LinearModel(n_dims = sigs.shape[2]).to(DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=SVM_LR)
    for epoch in range(N_SVM_EPOCHS):
        optimizer.zero_grad()
        background_MV = network(bkgs)
        signal_MV = network(sigs)
        signal_MV = torch.min(signal_MV, dim=1)[0] #second index are the indicies
        zero = torch.tensor(0).to(DEVICE)

        background_loss = torch.maximum(
                            zero,
                            1-background_MV).mean()
        signal_loss = torch.maximum(
                            zero,
                            1+signal_MV).mean()
        
        loss = background_loss + signal_loss
        if epoch % 50 == 0:
            #rint(network.layer_normal.weight.data.cpu().numpy()[0])
            print(loss.item())
        loss.backward()
        optimizer.step()

    torch.save(network.state_dict(), "./fm_model_quiet.pt")
    return network
    return network.layer.weight.data.cpu().numpy()[0]


def main(args):
    '''
    Fit a hyperplane to distinguish background, signal classes

    signals: shape (N_samples, time_axis, 5)
    backgrounds: shape (time_axis, 5)
    '''
    #np.save(args.save_file, np.array([0, -1, 0, 0, 0]))
    #return None
    norm_factors = []
    norm_factors_path = args.norm_factor_path
    if type(args.norm_factor_path) == str:
        norm_factors_path = [args.norm_factor_path]
    for file_name in norm_factors_path:
        norm_factors.append(np.load(f'{file_name}'))

    norm_factors = np.array(norm_factors)
    means = np.mean(norm_factors[:, 0, 0, :], axis=0)
    stds = np.mean(norm_factors[:, 1, 0, :], axis=0)


    trained_already = True
    if not trained_already:
        signal_evals = []
        if type(args.signal_path) == str:
            args.signal_path = [args.signal_path]
        midp_map = {'bbh_varying_snr_evals.npy': 1440,
    'sg_varying_snr_evals.npy': 1440,
    'wnb_varying_snr_evals.npy': 1850,
    'wnblf_varying_snr_evals.npy': 1850,
    'supernova_varying_snr_evals.npy': 2150}
        for file_name in args.signal_path:
            mid = midp_map[file_name.split("/")[-1]]
            signal_evals.append(np.load(f'{file_name}')[:, mid-150:mid+150, :])#[:1000]
        
        signal_evals = np.concatenate(signal_evals, axis=0)

        timeslide_evals = []
        timeslide_path = args.timeslide_path
        if type(args.timeslide_path) == str:
            timeslide_path = [args.timeslide_path]
        for file_name in timeslide_path:
            timeslide_evals.append(np.load(f'{file_name}'))

        
        
        timeslide_evals = np.concatenate(timeslide_evals, axis=0)
        signal_evals = (signal_evals-means)/stds
        timeslide_evals = (timeslide_evals-means)/stds

            
        
        #np.save(args.norm_factor_save_file, np.stack([means, stds], axis=0))
        network = optimize_hyperplane(signal_evals, timeslide_evals)
    else:
        network = LinearModel(n_dims = 1).to(DEVICE)
        network.load_state_dict(torch.load("./fm_model_quiet.pt", map_location=GPU_NAME))
        #np.save(args.save_file, optimal_coeffs)

    #print("network", network)
    #assert 0

    #do the timeslide analysis
    timeslide_folders = ["/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/lstm/timeslides_0",
                         "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/lstm/timeslides_1",
                         "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/lstm/timeslides_2"]
    save_path_bins = "./quiet_far_bins.npy"
    n_bins = 2*int(HISTOGRAM_BIN_MIN/HISTOGRAM_BIN_DIVISION)
    hist = np.zeros(n_bins)
    np.save(save_path_bins, hist)
    counter = 0
    for folder in timeslide_folders:
        for file in os.listdir(folder):
            if file[0] == "t": #is timeslide_evals
                evals = (np.load(f"{folder}/{file}") - means) / stds
                evals = network(torch.from_numpy(evals).float().to(DEVICE)).detach()
                #print("evals", evals)
                print(f"number done: {counter}", end = "\r")
                update = torch.histc(evals, bins=n_bins, 
                                 min=-HISTOGRAM_BIN_MIN, max=HISTOGRAM_BIN_MIN)
                past_hist = np.load(save_path_bins)
                #print("total filled:", np.sum(past_hist))
                new_hist = past_hist + update.cpu().numpy()
                np.save(save_path_bins, new_hist)
                counter += 1



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('save_file', type=str,
        help='Where to save the best final metric parameters')
    parser.add_argument('norm_factor_save_file', type=str,
        help='Where to save the normalization factors')
    parser.add_argument('--timeslide-path', type=str,
         nargs = '+', help='list[str] pointing to timeslide files ')
    parser.add_argument('--signal-path', type=str,
        nargs= '+', help='list[str] pointing to signal files')
    parser.add_argument('--norm-factor-path', type=str,
        nargs= '+', help='list[str] pointing to norm factors')
    
    args = parser.parse_args()
    main(args)
