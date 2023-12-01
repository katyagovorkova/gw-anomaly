import os
import numpy as np
import argparse
import torch
import time
from collections import defaultdict
from helper_functions import mae_torch, freq_loss_torch
from models import LSTM_AE, LSTM_AE_SPLIT, DUMMY_CNN_AE, FAT
import h5py
import sys
sys.path.append('/n/home00/emoreno/gw-anomaly/ml4gw')
from ml4gw.transforms import ShiftedPearsonCorrelation

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (NUM_IFOS,
                    SEG_NUM_TIMESTEPS,
                    BOTTLENECK,
                    MODEL,
                    FACTOR,
                    GPU_NAME,
                    RECREATION_LIMIT,
                    SAMPLE_RATE,
                    BATCH_SIZE,
                    SEG_NUM_TIMESTEPS,
                    SEGMENT_OVERLAP)

STRIDE = (SEG_NUM_TIMESTEPS - SEGMENT_OVERLAP) / SAMPLE_RATE

def data_iterator(dataset: h5py.Dataset, shifts: list[float], batch_size: int = BATCH_SIZE):
    shift_sizes = [int(i * SAMPLE_RATE) for i in shifts]
    num_channels, size = dataset.shape

    size -= max(shift_sizes)
    stride_size = int(STRIDE * SAMPLE_RATE)
    num_updates = size // stride_size
    num_batches = int(num_updates // batch_size)

    update_size = stride_size * batch_size
    idx = np.arange(update_size)
    x = np.zeros((num_channels, update_size))
    for i in range(num_batches):
        for j in range(num_channels):
            start = i + update_size + shift_sizes[j]
            stop = start + update_size
            x[j] = dataset[j, start: stop]
        yield torch.Tensor(x)

@torch.no_grad()
def quak_eval_snapshotter(data, models, batcher, device, SHIFT):
    pearson = ShiftedPearsonCorrelation(int(0.01 * SAMPLE_RATE))
    DEVICE = device

    for segment, dataset in data.items():
        start_time = time.time()
        # initialize a container for our predictions
        # and create an initial blank snapshot state.
        # The most efficient way to do this would be
        # to allocate all the memory up front since
        # we know how many steps to take, but this
        # will work for the time being.
        predictions = defaultdict(list)
        state = batcher.get_initial_state().to(DEVICE)
        num_preds = 0
        for x in data_iterator(dataset, SHIFT):
            # move the timeseries onto the GPU, then update our
            # state and perform preprocessing
            x = x.to(DEVICE)
            X, state = batcher(x, state)

            # feed the preprocessed data through each one of our models
            for name, model in models.items():
                
                y = model(X)
                loss = freq_loss_torch(y, X)
                predictions[name].append(loss)
            # compute the pearson correlation between
            # both interferometer channels
            corr = pearson(X[:, :1], X[:, 1:])
            corr = corr.max(dim=0).values[:, 0]
            predictions["pearson"].append(corr)

            num_preds += len(X)

        # concatenate everything back on the CPU then click our stopwatch
        predictions = {k: torch.cat(v) for k, v in predictions.items()}
        end_time = time.time()
        duration = end_time - start_time
        throughput = num_preds * STRIDE / duration
        print(f"Number of predictions: {num_preds}")
        print(f"Throughput: {throughput:.2f} Hz")
    
    return predictions


@torch.no_grad()
def quak_eval(data, model_path, device, reduce_loss=True, loaded_models=None):
    # data required to be torch tensor at this point

    # check if the evaluation has to be done for one model or for several
    loss = dict()
    if not reduce_loss:
        loss['original'] = dict()
        loss['recreated'] = dict()
        loss['loss'] = dict()
        loss['freq_loss'] = dict()

    for dpath in model_path:
        #coherent_loss = False
        #if dpath.split("/")[-1] in ['bbh.pt', 'sglf.pt', 'sghf.pt']:
        #    coherent_loss = True
        if loaded_models is None:
            

            model_name = dpath.split("/")[-1].split(".")[0]
            if MODEL[model_name] == "lstm":
                model = LSTM_AE_SPLIT(num_ifos=NUM_IFOS,
                                    num_timesteps=SEG_NUM_TIMESTEPS,
                                    BOTTLENECK=BOTTLENECK[model_name]).to(device)
            elif MODEL[model_name] == "dense":
                model = FAT(num_ifos=NUM_IFOS,
                            num_timesteps=SEG_NUM_TIMESTEPS,
                            BOTTLENECK=BOTTLENECK[model_name]).to(device)

            model.load_state_dict(torch.load(dpath, map_location=GPU_NAME))
        else:
            model = loaded_models[dpath]
        if reduce_loss:
            #if coherent_loss:
            loss[os.path.basename(dpath)[:-3]] = \
                freq_loss_torch(data, model(data).detach())
            #elif not coherent_loss:
            #    loss[os.path.basename(dpath)[:-3]] = \
            #        freq_loss_torch(data, model(data).detach())
        elif not reduce_loss:
            if coherent_loss:
                loss['loss'][os.path.basename(dpath)[:-3]] = \
                    mae_torch(data, model(
                        data).detach()).cpu().numpy()
                loss['freq_loss'][os.path.basename(dpath)[:-3]] = \
                    freq_loss_torch(data, model(data).detach())

            elif not coherent_loss:
                loss['loss'][os.path.basename(dpath)[:-3]] = \
                    mae_torch(data, model(
                        data).detach()).cpu().numpy()
                loss['freq_loss'][os.path.basename(dpath)[:-3]] = \
                    freq_loss_torch(data, model(data).detach())
            loss['original'][os.path.basename(
                dpath)[:-3]] = data[:RECREATION_LIMIT].cpu().numpy()
            loss['recreated'][os.path.basename(
                dpath)[:-3]] = model(data[:RECREATION_LIMIT]).detach().cpu().numpy()
    return loss


def main(args):

    DEVICE = torch.device(GPU_NAME)

    # load the data
    data = np.load(args.test_data)['data']
    data = torch.from_numpy(data).float().to(DEVICE)
    loss = quak_eval(data, args.model_path, DEVICE,
                     reduce_loss=args.reduce_loss)

    if args.reduce_loss:
        # move to CPU
        for key in loss.keys():
            loss[key] = loss[key].cpu().numpy()

    if args.save_file:
        np.savez(args.save_file, **loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        'test_data', help='Required path to the test data file')
    parser.add_argument('save_file', help='Required path to save the file to',
                        type=str)
    parser.add_argument('reduce_loss', help='Whether to reduce to loss values or return recreation',
                        type=str, default="False")

    parser.add_argument('--model-path', help='Required path to trained model',
                        nargs='+', type=str)
    args = parser.parse_args()
    args.reduce_loss = args.reduce_loss == "True"

    main(args)
