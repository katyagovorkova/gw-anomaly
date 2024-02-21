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
import setGPU
from tqdm import tqdm

#from models import (LSTM_AE, TransformerAutoencoder)
from models import TranAD_Basic, TranAD
from models_old import LSTM_AE

import sys
import os.path
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
    SEG_NUM_TIMESTEPS)

def convert_to_windows(data, w_size = 100):
	windows = []
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def main(args):
    TranAD = False
    if TranAD:
        # read the input data
        data = np.load(args.data)
        data = np.transpose(data, (0, 2, 1))
        print(data.shape)
        #data = data.reshape(-1, SEG_NUM_TIMESTEPS, NUM_IFOS)
        #data = np.transpose(data, (0, 2, 1))

        data = data.reshape(-1, 2)
        print(data.shape)

        # pick a random GPU device to train model on
        N_GPUs = torch.cuda.device_count()
        chosen_device = np.random.randint(0, N_GPUs)
        device = torch.device(f"cuda:{chosen_device}")
        if TRAINING_VERBOSE:
            print(f"Using device {device}")

        # create the model
        #AE = LSTM_AE(num_ifos=NUM_IFOS, 
        #            num_timesteps=SEG_NUM_TIMESTEPS,
        #            BOTTLENECK=BOTTLENECK,
        #            FACTOR=FACTOR).to(device)

        AE = TranAD(NUM_IFOS).to(device)

        optimizer = optim.Adam(AE.parameters())
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        if LOSS == "MAE":
            loss_fn = nn.L1Loss()
        else:
            # add in support for more losses?
            raise Exception("Unknown loss function")
        
        #loss_fn = nn.MSELoss()
        
        # create the dataset and validation set
        validation_split_index = int(VALIDATION_SPLIT * len(data))
        train_data = data[validation_split_index:]
        validation_data = data[:validation_split_index]

        train_data = torch.from_numpy(train_data).float().to(device)
        validation_data = torch.from_numpy(validation_data).float().to(device)

        print(train_data.shape)
        train_data = convert_to_windows(train_data, w_size=100)
        validation_data = convert_to_windows(validation_data, w_size=100)
        print('converted to windows')
        print(data.shape)
        dataloader = []
        dataloader_val = []
        N_batches = len(train_data) // BATCH_SIZE
        for i in range(N_batches-1):
            start, end = i*BATCH_SIZE, (i+1) * BATCH_SIZE
            dataloader.append(train_data[start:end])

        N_batches_val = len(validation_data) // BATCH_SIZE
        validation_data = validation_data[:int((N_batches_val-1) * BATCH_SIZE)]
        for i in range(N_batches_val-1):
            start, end = i*BATCH_SIZE, (i+1) * BATCH_SIZE
            dataloader_val.append(validation_data[start:end])

        training_history = {
            'train_loss': [],
            'val_loss': []
        }
        print('starting training. training shape: ', train_data.shape, 'validation shape: ', validation_data.shape)

        # training loop
        
        for epoch_num in range(EPOCHS):
            ts = time.time()
            epoch_train_loss = 0
            AE.train()
            with tqdm(dataloader, unit="batch") as pbar:
                for data in pbar:
                    optimizer.zero_grad()
                    local_bs = data.shape[0]
                    window = data.permute(1, 0, 2)
                    feats = window.shape[2]
                    elem = window[-1, :, :].view(1, local_bs, feats)
                    output = AE(window, elem)
                    #print(output.shape)
                    #print(elem.shape)
                    loss = loss_fn(output, elem) if not isinstance(output, tuple) else (1 / (epoch_num+1)) * loss_fn(output[0], elem) + (1 - 1/(epoch_num+1)) * loss_fn(output[1], elem)
                    del window
                    #print(output)
                    epoch_train_loss += loss.item()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                
            epoch_train_loss /= N_batches
            training_history['train_loss'].append(epoch_train_loss)
            AE.eval()
            validation_pred0 = []
            validation_pred1 = []
            validation_elem = []
            with torch.no_grad():
                with tqdm(dataloader_val, unit="batch") as pbar:
                    for data in pbar:
                        local_bs = data.shape[0]
                        window = data.permute(1, 0, 2)
                        feats = window.shape[2]
                        elem = window[-1, :, :].view(1, local_bs, feats)
                        output = AE(window, elem)
                        validation_pred0.append(output[0])
                        validation_pred1.append(output[1])
                        validation_elem.append(elem)
            
            validation_pred0 = torch.cat(validation_pred0, dim=0)
            validation_pred1 = torch.cat(validation_pred1, dim=0)
            validation_elem = torch.cat(validation_elem, dim=0)
            validation_loss = (1 / (epoch_num+1)) * loss_fn(validation_pred0, validation_elem) + (1 - 1/(epoch_num+1)) * loss_fn(validation_pred1, validation_elem)
            training_history['val_loss'].append(validation_loss.item())
            scheduler.step(validation_loss)

            if TRAINING_VERBOSE:
                elapsed_time = time.time() - ts
                data_name = (args.data).split("/")[-1][:-4]
                print(f"data: {data_name}, epoch: {epoch_num}, train loss: {epoch_train_loss :.4f}, val loss: {validation_loss :.4f}, time: {elapsed_time :.4f}")
    else: 
        # read the input data
        data = np.load(args.data)
        data = np.transpose(data, (0, 2, 1))
        data = data.reshape(-1, SEG_NUM_TIMESTEPS, NUM_IFOS)
        #data = np.transpose(data, (0, 2, 1))
        
        # pick a random GPU device to train model on
        N_GPUs = torch.cuda.device_count()
        chosen_device = np.random.randint(0, N_GPUs)
        device = torch.device(f"cuda:{chosen_device}")
        if TRAINING_VERBOSE:
            print(f"Using device {device}")

        # create the model
        AE = LSTM_AE(num_ifos=NUM_IFOS, 
                    num_timesteps=SEG_NUM_TIMESTEPS,
                    BOTTLENECK=BOTTLENECK,
                    FACTOR=FACTOR).to(device)


        #AE = CONV_LSTM_AE(input_dims=(1024, 2), encoding_dim=BOTTLENECK,
        #                  kernel=(2, 1), stride=(2, 2),
        #                   h_conv_channels=[4, 4], h_lstm_channels=[16, 16]).to(device)
        

        optimizer = optim.Adam(AE.parameters())
        scheduler = ReduceLROnPlateau(optimizer, 'min')
        if LOSS == "MAE":
            loss_fn = nn.L1Loss()
        else:
            # add in support for more losses?
            raise Exception("Unknown loss function")
        
        #loss_fn = nn.MSELoss()
        
        # create the dataset and validation set
        validation_split_index = int(VALIDATION_SPLIT * len(data))
        train_data = data[validation_split_index:]
        validation_data = data[:validation_split_index]

        train_data = torch.from_numpy(train_data).float().to(device)
        validation_data = torch.from_numpy(validation_data).float().to(device)

        dataloader = []
        dataloader_val = []
        N_batches = len(train_data) // BATCH_SIZE
        for i in range(N_batches-1):
            start, end = i*BATCH_SIZE, (i+1) * BATCH_SIZE
            dataloader.append(train_data[start:end])

        N_batches_val = len(validation_data) // BATCH_SIZE
        validation_data = validation_data[:int((N_batches_val-1) * BATCH_SIZE)]
        for i in range(N_batches_val-1):
            start, end = i*BATCH_SIZE, (i+1) * BATCH_SIZE
            dataloader_val.append(validation_data[start:end])

        training_history = {
            'train_loss': [],
            'val_loss': []
        }
        print('starting training. training shape: ', train_data.shape, 'validation shape: ', validation_data.shape)

        # training loop
        
        for epoch_num in range(EPOCHS):
            ts = time.time()
            epoch_train_loss = 0
            AE.train()
            with tqdm(dataloader, unit="batch") as pbar:
                for data in pbar:
                    optimizer.zero_grad()
                    output = AE(data)
                    loss = loss_fn(output, data) 
                    del data
                    #print(output)
                    epoch_train_loss += loss.item()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                
            epoch_train_loss /= N_batches
            training_history['train_loss'].append(epoch_train_loss)
            AE.eval()
            validation_pred = []
            with torch.no_grad():
                with tqdm(dataloader_val, unit="batch") as pbar:
                    for data in pbar:
                        output = AE(data)
                        validation_pred.append(output)

            
            validation_pred = torch.cat(validation_pred, dim=0)
            validation_loss = loss_fn(validation_pred, validation_data)
            training_history['val_loss'].append(validation_loss.item())
            scheduler.step(validation_loss)

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
    args = parser.parse_args()
    main(args)
