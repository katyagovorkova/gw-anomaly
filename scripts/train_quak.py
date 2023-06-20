import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from scipy.linalg import dft

from models import (
    LSTM_AE,
    LSTM_AE_ERIC,
    DUMMY_CNN_AE,
    FAT)

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    BOTTLENECK,
    FACTOR,
    EPOCHS,
    BATCH_SIZE,
    LOSS,
    OPTIMIZER,
    VALIDATION_SPLIT,
    TRAINING_VERBOSE,
    NUM_IFOS,
    SEG_NUM_TIMESTEPS,
    GPU_NAME,
    LIMIT_TRAINING_DATA)
DEVICE = torch.device(GPU_NAME)

def main(args):
    # read the input data
    data = np.load(args.data)
    print(f'loaded data shape is {data.shape}')
    if LIMIT_TRAINING_DATA is not None:
        data = data[:LIMIT_TRAINING_DATA]
    # create the model
    if args.model=='dense':
        AE = FAT(
            num_ifos=NUM_IFOS,
            num_timesteps=SEG_NUM_TIMESTEPS,
            BOTTLENECK=BOTTLENECK,
            FACTOR=FACTOR).to(DEVICE)
    elif args.model=='lstm':
        AE = LSTM_AE(
            input_dim=NUM_IFOS,
            encoding_dim=10,
            h_dims=[64],
        )
    elif args.model=='transformer':
        AE = None
        print('OOOPS NOT IMPLEMENTED')

    optimizer = optim.Adam(AE.parameters())

    if LOSS == "MAE":
        loss_fn = nn.L1Loss()
    else:
        # add in support for more losses?
        raise Exception("Unknown loss function")

    # create the dataset and validation set
    validation_split_index = int((1-VALIDATION_SPLIT) * len(data))
    train_data = data[:validation_split_index]
    validation_data = data[validation_split_index:]

    train_data = torch.from_numpy(train_data).float().to(DEVICE)
    print(f'training data shape is {train_data.shape}')
    validation_data = torch.from_numpy(validation_data).float().to(DEVICE)

    dataloader = []
    N_batches = len(train_data) // BATCH_SIZE
    for i in range(N_batches-1):
        start, end = i*BATCH_SIZE, (i+1) * BATCH_SIZE
        dataloader.append(train_data[start:end])

    training_history = {
        'train_loss': [],
        'val_loss': []
    }
    # training loop

    for epoch_num in range(EPOCHS):
        ts = time.time()
        epoch_train_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            #print("shape 83", batch.shape)
            output = AE(batch)
            loss = loss_fn(batch, output)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_train_loss /= N_batches
        training_history['train_loss'].append(epoch_train_loss)
        validation_loss = loss_fn(validation_data,
                                AE(validation_data))
        training_history['val_loss'].append(validation_loss.item())
        #scheduler.step(validation_loss)

        if TRAINING_VERBOSE:
            elapsed_time = time.time() - ts
            data_name = (args.data).split("/")[-1][:-4]
            print(f"data: {data_name}, epoch: {epoch_num}, train loss: {epoch_train_loss :.4f}, val loss: {validation_loss :.4f}, time: {elapsed_time :.4f}")

    # save the model
    torch.save(AE.state_dict(), f'{args.save_file}')

    # save training history
    np.save(f'{args.savedir}/loss_hist.npy',
            np.array(training_history['train_loss']))
    np.save(f'{args.savedir}/val_loss_hist.npy',
            np.array(training_history['val_loss']))

    # plot training history
    plt.figure(figsize=(15, 10))
    plt.plot(np.array(training_history['train_loss']), label='loss')
    plt.plot(np.array(training_history['val_loss']), label='val loss')
    plt.legend()
    plt.xlabel('Number of epochs', fontsize=17)
    plt.ylabel('Loss', fontsize=17)
    plt.title('Loss curve for training network', fontsize=17)
    plt.savefig(f'{args.savedir}/loss.pdf', dpi=300)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data', help='Input dataset',
        type=str)
    parser.add_argument('save_file', help='Where to save the trained model',
        type=str)
    parser.add_argument('savedir', help='Where to save the plots',
        type=str)

    parser.add_argument('--model', help='Required path to trained model',
                        type=str, choices=['lstm', 'dense', 'transformer'])

    args = parser.parse_args()
    main(args)
