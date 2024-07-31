import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import matplotlib.pyplot as plt

from config import (
    FM_TIMESLIDE_TOTAL_DURATION,
    TIMESLIDE_TOTAL_DURATION,
    SAMPLE_RATE,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN,
    RETURN_INDIV_LOSSES,
    FACTORS_NOT_USED_FOR_FM,
    SMOOTHING_KERNEL_SIZES,
    SUPERVISED_BKG_TIMESLIDE_LEN,
    SUPERVISED_N_BKG_SAMPLE,
    SUPERVISED_REDUCE_N_BKG,
    SUPERVISED_BATCH_SIZE,
    SUPERVISED_VALIDATION_SPLIT,
    SUPERVISED_EPOCHS,
    TRAINING_VERBOSE,
)

class SupervisedModel(nn.Module):

    def __init__(self, n_dims):
        super(SupervisedModel, self).__init__()
        self.layer1 = nn.Linear(400, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return F.sigmoid(self.layer4(x))


def make_bkg_from_timeslides(data_path, gpu):
    DEVICE = torch.device(f'cuda:{gpu}')
    data = np.load(data_path)['data']
    data = torch.from_numpy(data).to(DEVICE)

    reduction = 20  # for things to fit into memory nicely

    timeslide_total_duration = SUPERVISED_BKG_TIMESLIDE_LEN

    sample_length = data.shape[1] / SAMPLE_RATE
    n_timeslides = int(timeslide_total_duration //
                        sample_length) * reduction
    print('Number of timeslides:', n_timeslides)


    bkg_datae = []
    for timeslide_num in range(1, n_timeslides + 1):
        print(f'starting timeslide: {timeslide_num}/{n_timeslides}')

        indicies_to_slide = np.random.uniform(
            SAMPLE_RATE, data.shape[1] - SAMPLE_RATE)
        indicies_to_slide = int(indicies_to_slide)
        timeslide = torch.empty(data.shape, device=DEVICE)

        # hanford unchanged
        timeslide[0, :] = data[0, :]

        # livingston slid
        timeslide[1, :indicies_to_slide] = data[1, -indicies_to_slide:]
        timeslide[1, indicies_to_slide:] = data[1, :-indicies_to_slide]


         # make a random cut with the reduced shape
        reduced_len = int(data.shape[1] / reduction)
        start_point = int(np.random.uniform(
            0, data.shape[1] - SAMPLE_RATE - reduced_len))
        timeslide = timeslide[:, start_point:start_point + reduced_len]

        timeslide = timeslide[:, :(timeslide.shape[1] // 1000) * 1000] #(2, len)
        
        #timeslide_datae.append(timeslide)
        for _ in range(SUPERVISED_N_BKG_SAMPLE):

            # sample a window for the background dataset
            start = np.random.uniform(0, timeslide.shape[1]-200)
            bkg_datae.append(timeslide[:, int(start):int(start)+200])


    bkg_datae = torch.cat(bkg_datae, dim=0)
    print("N SAMPLES", len(bkg_datae))
    print("REDUCE TO 100,000")
    p = torch.randperm(len(bkg_datae), device=DEVICE)
    bkg_datae = bkg_datae[p][:SUPERVISED_REDUCE_N_BKG]

    return bkg_datae #(N_data, 2, 200)




def main(args):
    DEVICE = torch.device(f'cuda:{args.gpu}')
    bkg_data = make_bkg_from_timeslides(args.data_path[0], args.gpu) #(N_data, 2, 200)
    bkg_data = torch.reshape(bkg_data, (bkg_data.shape[0], bkg_data.shape[1]*bkg_data.shape[2]))

    data_name = (args.data).split('/')[-1][:-4]
    assert data_name in ['bbh', 'sglf', 'sghf']

    # curriculum learning scheme
    # n_currics, n_samples, ifo, timesteps
    noisy_data = np.load(args.data)['noisy']
    clean_data = np.load(args.data)['clean']

    n_currics = len(noisy_data)
    print('n_currics', n_currics)
    # normalization scheme

    stds = np.std(noisy_data, axis=-1)[:, :, :, np.newaxis]
    noisy_data = noisy_data / stds
    clean_data = clean_data / stds

    # shuffle
    p = np.random.permutation(noisy_data.shape[1])
    noisy_data = noisy_data[:, p, :, :]
    clean_data = clean_data[:, p, :, :]

    model = SupervisedModel()

    # the training itself
    #for c in range(n_currics):
    c = len(n_currics) - 1 #only use the last curriculum
    # noisy_data, clean_data
    data_x, data_y = noisy_data[c], clean_data[c]
    data_x_last, data_y_last = noisy_data[-1], clean_data[-1]

    # create the dataset and validation set
    validation_split_index = int((1 - SUPERVISED_VALIDATION_SPLIT) * len(data_x))

    train_data_x = data_x[:validation_split_index]
    train_data_x = torch.from_numpy(train_data_x).float().to(DEVICE)
    train_data_y = data_y[:validation_split_index]
    train_data_y = torch.from_numpy(train_data_y).float().to(DEVICE)

    validation_data_x = data_x_last[validation_split_index:]
    validation_data_x = torch.from_numpy(
        validation_data_x).float().to(DEVICE)
    validation_data_y = data_y_last[validation_split_index:]
    validation_data_y = torch.from_numpy(
        validation_data_y).float().to(DEVICE)

    dataloader = []
    N_batches = len(train_data_x) // SUPERVISED_BATCH_SIZE
    for i in range(N_batches - 1):
        start, end = i * SUPERVISED_BATCH_SIZE, (i + 1) * SUPERVISED_BATCH_SIZE
        dataloader.append([train_data_x[start:end],
                            train_data_y[start:end]])

    # dataloader, validation_data_x, validation_data_y
    epoch_count = 0

    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    training_history = {
        'train_loss': [],
        'val_loss': []    }
    
    train_signal_x = train_data_x
    train_signal_x = torch.reshape(train_signal_x, (train_signal_x.shape[0], train_signal_x.shape[1]*train_signal_x.shape[2] ))
    val_signal_x = validation_data_x
    val_signal_x = torch.reshape(val_signal_x, (val_signal_x.shape[0], val_signal_x.shape[1]*val_signal_x.shape[2] ))

    bkg_split_index = int(len(bkg_data)*SUPERVISED_VALIDATION_SPLIT)
    train_bkg_x = bkg_data[bkg_split_index:]
    val_bkg_x = bkg_data[:bkg_split_index]

    # make y labels
    train_signal_y = torch.ones(train_signal_x.shape[0])
    val_signal_y = torch.ones(val_signal_x.shape[0])

    train_bkg_y = torch.zeros(train_bkg_x.shape[0])
    val_bkg_y = torch.zeros(val_bkg_x.shape[0])


    train_x = torch.cat([train_signal_x, train_bkg_x], dim=0)
    val_x = torch.cat([val_signal_x, val_bkg_x], dim=0)

    train_y = torch.cat([train_signal_y, train_bkg_y], dim=0)
    val_y = torch.cat([val_signal_y, val_bkg_y], dim=0)

    p = torch.randperm(len(train_x), device=DEVICE)
    train_x = train_x[p]
    train_y = train_y[p]

    dataloader = []
    N_batches = len(train_x) // SUPERVISED_BATCH_SIZE
    for i in range(N_batches - 1):
        start, end = i * SUPERVISED_BATCH_SIZE, (i + 1) * SUPERVISED_BATCH_SIZE
        dataloader.append([train_x[start:end],
                            train_y[start:end]])

    for epoch_num in range(SUPERVISED_EPOCHS):
        epoch_count += 1
        ts = time.time()
        epoch_train_loss = 0
        for batch in dataloader:
            train_x_batch, train_y_batch = batch
            optimizer.zero_grad()

            output = model(train_x_batch)
            loss = loss_fn(output, train_y_batch)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_train_loss /= N_batches
        training_history['train_loss'].append(epoch_train_loss)
        validation_loss = loss_fn(model(val_x), val_y)

        training_history['val_loss'].append(validation_loss.item())

        if TRAINING_VERBOSE:
            elapsed_time = time.time() - ts

            print(f'data: {data_name}, epoch: {epoch_count}, train loss: {epoch_train_loss :.4f}, val loss: {validation_loss :.4f}, time: {elapsed_time :.4f}')

        torch.save(model.state_dict(), f'{args.save_file}')

        # save training history
        np.save(f'{args.savedir}/loss_hist.npy',
                np.array(training_history['train_loss']))
        np.save(f'{args.savedir}/val_loss_hist.npy',
                np.array(training_history['val_loss']))


        fig, ax = plt.subplots(1, figsize=(8, 5))
        epochs = np.linspace(1, epoch_count, epoch_count)

        ax.plot(epochs, np.array(training_history[
                'train_loss']), label='Training loss')
        ax.plot(epochs, np.array(training_history[
                'val_loss']), label='Validation loss')

        ax.legend()
        ax.set_xlabel('Epochs', fontsize=15)
        ax.set_ylabel('Loss', fontsize=15)
        ax.grid()

        plt.savefig(f'{args.savedir}/loss.pdf', dpi=300)







if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data', type=str,
                        help='Input signal dataset')
    parser.add_argument('data-path', type=str, nargs='+',
                        help='Directory containing the timeslides')
    parser.add_argument('save-file', type=str,
                        help='Where to save the trained model')
    parser.add_argument('savedir', type=str,
                        help='Where to save the loss plots')
    # Additional argument

    parser.add_argument('--gpu', type=str, default='1',
                        help='On which GPU to run')

    

    args = parser.parse_args()
    #args.fm_shortened_timeslides = args.fm_shortened_timeslides == 'True'

    main(args)
