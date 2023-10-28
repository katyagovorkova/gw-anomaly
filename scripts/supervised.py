import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import matplotlib.pyplot as plt

import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from config import (
    SEG_NUM_TIMESTEPS,
    SEGMENT_OVERLAP,
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
    SUPERVISED_FAR_TIMESLIDE_LEN,
    SUPERVISED_SMOOTHING_KERNEL,
    HISTOGRAM_BIN_DIVISION,
    HISTOGRAM_BIN_MIN,
    SNR_VS_FAR_BAR,
    VARYING_SNR_LOW,
    VARYING_SNR_HIGH,
    SNR_VS_FAR_HL_LABELS,
    SNR_VS_FAR_HORIZONTAL_LINES,
    SUPERVISED_FAR_TIMESLIDE_LEN
)
from helper_functions import (far_to_metric)
from labellines import labelLines

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class SupervisedModel(nn.Module):

    def __init__(self, seq_len, n_features):
        super(SupervisedModel, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.rnn1_0 = nn.LSTM(
            input_size=1,
            hidden_size=4,
            num_layers=2,
            batch_first=True
        )
        self.rnn1_1 = nn.LSTM(
            input_size=1,
            hidden_size=4,
            num_layers=2,
            batch_first=True
        )

        self.encoder_dense_scale = 20
        self.linear1 = nn.Linear(
            in_features=2**8, out_features=self.encoder_dense_scale * 4)
        self.linear2 = nn.Linear(
            in_features=self.encoder_dense_scale * 4, out_features=self.encoder_dense_scale * 2)
        self.linear_passthrough = nn.Linear(
            2 * seq_len, self.encoder_dense_scale * 2)
        self.linear3 = nn.Linear(
            in_features=self.encoder_dense_scale * 4, out_features=21-len(FACTORS_NOT_USED_FOR_FM))

        self.linearH = nn.Linear(4 * seq_len, 2**7)
        self.linearL = nn.Linear(4 * seq_len, 2**7)

        self.layer1 = nn.Linear(21-len(FACTORS_NOT_USED_FOR_FM), 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, 2 * self.seq_len)
        other_dat = self.linear_passthrough(x_flat)
        Hx, Lx = x[:, 0, :][:, :, None], x[:, 1, :][:, :, None]

        Hx, (_, _) = self.rnn1_0(Hx)
        Hx = Hx.reshape(batch_size, 4 * self.seq_len)
        Hx = F.tanh(self.linearH(Hx))

        Lx, (_, _) = self.rnn1_1(Lx)
        Lx = Lx.reshape(batch_size, 4 * self.seq_len)
        Lx = F.tanh(self.linearL(Lx))

        x = torch.cat([Hx, Lx], dim=1)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = torch.cat([x, other_dat], axis=1)
        #print("216", x.shape)
        x = F.relu(self.linear3(x))

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return F.sigmoid(self.layer4(x))

# class SupervisedModel(nn.Module):

#     def __init__(self, n_dims):
#         super(SupervisedModel, self).__init__()
#         self.layer1 = nn.Linear(400, 160)
#         self.layer12 = nn.Linear(160, 32)
#         self.layer2 = nn.Linear(32, 32)
#         self.layer3 = nn.Linear(32, 32)
#         self.layer4 = nn.Linear(32, 1)

#     def forward(self, x):

#         x = F.relu(self.layer1(x))
#         # x = F.relu(self.layer11(x))
#         x = F.relu(self.layer12(x))
#         x = F.relu(self.layer2(x))
#         x = F.relu(self.layer3(x))
#         return F.sigmoid(self.layer4(x))

def calculate_means(metric_vals, snrs, bar):
    # helper function for SNR vs FAR plot
    means, stds = [], []
    snr_plot = []

    for i in np.arange(VARYING_SNR_LOW, VARYING_SNR_HIGH, bar):

        points = []
        for shift in range(bar):
            for elem in np.where(((snrs - shift).astype(int)) == i)[0]:
                points.append(elem)
        if len(points) == 0:
            continue

        snr_plot.append(i + bar / 2)
        MV = []
        for point in points:
            MV.append(metric_vals[point])
        MV = np.array(MV)
        means.append(np.mean(MV))
        stds.append(np.std(MV))

    return np.array(snr_plot), np.array(means), np.array(stds)

def make_bkg_from_timeslides(data_path, gpu,
                             timeslide_total_duration=SUPERVISED_BKG_TIMESLIDE_LEN,
                             return_raw_timeslide=False):

    DEVICE = torch.device(f'cuda:{gpu}')
    data = np.load(data_path)['data']
    data = torch.from_numpy(data).to(DEVICE)

    reduction = 20  # for things to fit into memory nicely

    #timeslide_total_duration =
    sample_length = data.shape[1] / SAMPLE_RATE
    n_timeslides = int(timeslide_total_duration //
                        sample_length) * reduction
    print('Number of timeslides:', n_timeslides)

    bkg_datae = []
    timeslides = []
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
        if not return_raw_timeslide:
            for _ in range(SUPERVISED_N_BKG_SAMPLE):

                # sample a window for the background dataset
                start = np.random.uniform(0, timeslide.shape[1]-200)
                bkg_datae.append(timeslide[None, :, int(start):int(start)+200])
        else:
            timeslides.append(timeslide)

    if not return_raw_timeslide:
        bkg_datae = torch.cat(bkg_datae, dim=0)
        print("N SAMPLES", len(bkg_datae))
        print("REDUCE TO 100,000")
        p = torch.randperm(len(bkg_datae), device=DEVICE)
        bkg_datae = bkg_datae[p][:SUPERVISED_REDUCE_N_BKG]
        return bkg_datae #(N_data, 2, 200)
    else:
        #timeslides = torch.cat(timeslides, dim=0)
        print("Shape of concatenated timeslides: ", timeslides[0].shape)
        return timeslides

def inverter(x):
    # logit(1-x), map signals to zero and invert the sigmoid
    return torch.log((1-x)/(x))

def main(args):
    DEVICE = torch.device(f'cuda:{args.gpu}')
    if not os.path.exists(f'{args.save_file}'):

        bkg_data = make_bkg_from_timeslides(args.data_path, args.gpu) #(N_data, 2, 200)
        print(f'Shape: {bkg_data.shape}')
        # bkg_data = torch.reshape(bkg_data, (bkg_data.shape[0], bkg_data.shape[1]*bkg_data.shape[2]))

        data_name = (args.data).split('/')[-1][:-4]
        assert data_name in ['bbh', 'sglf', 'sghf']

        # curriculum learning scheme
        noisy_data = np.load(args.data)['noisy']

        n_currics = len(noisy_data)
        print('n_currics', n_currics)
        # normalization scheme
        stds = np.std(noisy_data, axis=-1)[:, :, :, np.newaxis]
        noisy_data = noisy_data / stds
        # shuffle
        p = np.random.permutation(noisy_data.shape[1])
        noisy_data = noisy_data[:, p, :, :]

        model = SupervisedModel(200,2).to(DEVICE)
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f'N parameters {count_parameters(model)}')
        # the training itself
        c = n_currics - 1 #only use the last curriculum
        data_x = noisy_data[-1]

        # create the dataset and validation set
        validation_split_index = int((1 - SUPERVISED_VALIDATION_SPLIT) * len(data_x))

        train_data_x = data_x[:validation_split_index]
        train_data_x = torch.from_numpy(train_data_x).float().to(DEVICE)

        validation_data_x = data_x[validation_split_index:]
        validation_data_x = torch.from_numpy(
            validation_data_x).float().to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        loss_fn = nn.BCELoss()

        training_history = {
            'train_loss': [],
            'val_loss': []    }

        train_signal_x = train_data_x
        # train_signal_x = torch.reshape(train_signal_x, (train_signal_x.shape[0], train_signal_x.shape[1]*train_signal_x.shape[2] ))
        val_signal_x = validation_data_x
        # val_signal_x = torch.reshape(val_signal_x, (val_signal_x.shape[0], val_signal_x.shape[1]*val_signal_x.shape[2] ))

        bkg_split_index = int(len(bkg_data)*SUPERVISED_VALIDATION_SPLIT)
        train_bkg_x = bkg_data[bkg_split_index:]
        val_bkg_x = bkg_data[:bkg_split_index]

        print(f'signal {train_data_x.shape}; background {bkg_data.shape}')

        # make y labels
        train_signal_y = torch.ones(train_signal_x.shape[0])
        val_signal_y = torch.ones(val_signal_x.shape[0])

        train_bkg_y = torch.zeros(train_bkg_x.shape[0])
        val_bkg_y = torch.zeros(val_bkg_x.shape[0])


        train_x = torch.cat([train_signal_x, train_bkg_x], dim=0)
        val_x = torch.cat([val_signal_x, val_bkg_x], dim=0)

        train_y = torch.cat([train_signal_y, train_bkg_y], dim=0)
        val_y = torch.cat([val_signal_y, val_bkg_y], dim=0)

        p = torch.randperm(len(train_x))
        train_x = train_x[p].to(DEVICE)
        train_y = train_y[p].to(DEVICE)

        dataloader = []
        N_batches = len(train_x) // SUPERVISED_BATCH_SIZE
        for i in range(N_batches - 1):
            start, end = i * SUPERVISED_BATCH_SIZE, (i + 1) * SUPERVISED_BATCH_SIZE
            dataloader.append([train_x[start:end],
                                torch.reshape(train_y[start:end], (-1,1) ) ])

        early_stopper = EarlyStopper(patience=3, min_delta=10)

        epoch_count = 0
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
            validation_loss = loss_fn(model(val_x), torch.reshape(val_y, (-1,1)).to(DEVICE))

            training_history['val_loss'].append(validation_loss.item())

            if TRAINING_VERBOSE:
                elapsed_time = time.time() - ts

                print(f'data: {data_name}, epoch: {epoch_count}, train loss: {epoch_train_loss :.4f}, val loss: {validation_loss :.4f}, time: {elapsed_time :.4f}')



            if early_stopper.early_stop(validation_loss):
                break


        # moving this outside - no need for it to be called every training epoch?? python complains about potential memory problems
        # due to opening so many figures
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

    else: # bypass having to train the model each time, just load it if it exists (has been trained) already
        model = SupervisedModel(200,2).to(DEVICE)
        model.load_state_dict(torch.load(f'{args.save_file}'))
        print(model)

    # RYAN EDITS BEGIN -
    # 1) evaluate on timeslides to get metric values for certain false alarm rates (let's say for 1/month, especially as everything is being run in 1 file)
    # 2) plot ROC & detection efficiency

    if not os.path.exists(f"{args.savedir}/far_hist.npy"):
        # create timeslides for FAR computation
        bkg_data = make_bkg_from_timeslides(args.data_path, args.gpu,
                                            timeslide_total_duration=SUPERVISED_FAR_TIMESLIDE_LEN,
                                            return_raw_timeslide=True) #(N, 734000) (length of 1 "slide" in datapoints)

        n_bins = 2 * int(HISTOGRAM_BIN_MIN / HISTOGRAM_BIN_DIVISION)
        hist = np.zeros(n_bins)

        DEVICE1 = torch.device("cuda:1")
        model.to(DEVICE1)
        # loop over each data for now, GPU memory stuff
        for n in range(len(bkg_data)):
            seg = bkg_data[n].to(DEVICE1)
            print(seg.shape)

            # window the segment
            seg = seg[:, :(seg.shape[1]//SEG_NUM_TIMESTEPS) *SEG_NUM_TIMESTEPS ].T
            segs = seg.unfold(dimension=0, size=SEG_NUM_TIMESTEPS, step=SEGMENT_OVERLAP)
            # segs shape (N, 200)
            evals = inverter(model(segs))[:, 0]

            # smoothing
            #evals = evals.detach().cpu().numpy()
            #evals = np.apply_along_axis(lambda m: np.convolve(m, np.ones(SUPERVISED_SMOOTHING_KERNEL)/SUPERVISED_SMOOTHING_KERNEL, mode='same'),
            #    axis=0,
            #    arr=evals)

            # do smoothing without converting to numpy - same structure as doing the windowing
            seg_smooth = evals.unfold(dimension=0, size=SUPERVISED_SMOOTHING_KERNEL, step=1)
            kernel = torch.ones((SUPERVISED_SMOOTHING_KERNEL, 1))/SUPERVISED_SMOOTHING_KERNEL
            seg_smooth = torch.matmul(seg_smooth, kernel.to(DEVICE1))#(N, 1)

            update = torch.histc(seg_smooth, bins=n_bins,
                                    min=-HISTOGRAM_BIN_MIN, max=HISTOGRAM_BIN_MIN)

            hist = hist + update.cpu().numpy()
            del evals

        np.save(f"{args.savedir}/far_hist.npy", hist)

    else:
        hist = np.load(f"{args.savedir}/far_hist.npy")

    # hist assembled by this point

    data_path = "/home/katya.govorkova/gw-anomaly/output/O3av2/data/"
    plotting_data = np.load(f"{data_path}/bbh_varying_snr.npz")['data']
    plotting_data = np.swapaxes(plotting_data, 1, 2)
    SNRs = np.load(f"{data_path}/bbh_varying_snr_SNR.npz.npy")

    plotting_data = torch.from_numpy(plotting_data).float().to(DEVICE)
    plotting_data = plotting_data.unfold(dimension=1, size=SEG_NUM_TIMESTEPS, step=SEGMENT_OVERLAP)

    # just do it one-by-one here, doesn't take that much time
    model = model.to(DEVICE)
    minvals = []
    for n in range(len(plotting_data)):
        eval = inverter(model(plotting_data[n])).detach().cpu().numpy()

        # numpy smoothing convolution, not many datapoints
        eval = np.apply_along_axis(lambda m: np.convolve(m, np.ones(SUPERVISED_SMOOTHING_KERNEL)/SUPERVISED_SMOOTHING_KERNEL, mode='valid'),
            axis=0,
            arr=eval)

        minval = np.min(eval)
        minvals.append(minval)

    fm_vals = np.array(minvals)
    amp_measure = SNRs
    far_hist = hist

    show_detec_effic = False
    if show_detec_effic:
        amp_measure_plot, means_plot, stds_plot = calculate_means(
                    fm_vals, amp_measure, bar=SNR_VS_FAR_BAR)

        fig, axs = plt.subplots(1, figsize=(12, 8))
        axs.plot(amp_measure_plot, means_plot, color="blue", label=f'BBH', linewidth=2)
        axs.fill_between(amp_measure_plot,
                        (means_plot) - stds_plot / 2,
                        (means_plot) + stds_plot / 2,
                        alpha=0.15,
                        color="blue")

        for i, label in enumerate(SNR_VS_FAR_HL_LABELS):
            metric_val_label = far_to_metric(
                SNR_VS_FAR_HORIZONTAL_LINES[i], far_hist)
            print(label, metric_val_label)
            if metric_val_label is not None:
                axs.axhline(y=metric_val_label, alpha=0.8**i, label=f'1/{label}', c='black')

        labelLines(axs.get_lines(), zorder=2.5, xvals=(
            15, 20, 15, 30, 35, 40, 25, 30, 35, 40, 45))

        axs.set_title("Supervised Detection Efficiency", fontsize=20)


        #axs.set_ylim(-40,0)

        plt.grid(True)
        fig.tight_layout()
        plt.savefig(f'{args.savedir}/bbh_detection_efficiency.pdf', dpi=300)
        plt.close()


    # make the ROC curve
    # fm_vals, amp_measure, far_hist

    fig, axs = plt.subplots(1, figsize=(12, 8))

    am_bins = np.arange(0, VARYING_SNR_HIGH, 1)
    # CHANGE FROM 1/HOUR!!
    print("CHANGE FROM 1/HOUR!!")
    axs.set_ylabel('Fraction of events detected at FAR 1/hour', fontsize=20)
    metric_val_label = far_to_metric(
            12*30*24*3600, far_hist)

    nbins = len(am_bins)
    am_bin_detected = [0]*nbins
    am_bin_total = [0]*nbins
    for i, am in enumerate(amp_measure):
        insert_location = np.searchsorted(am_bins, am)
        #print(am)
        if insert_location >= VARYING_SNR_HIGH:
            continue
        #print(insert_location)
        am_bin_total[insert_location] += 1
        detec_stat = fm_vals[i]
        if detec_stat <= metric_val_label:
            am_bin_detected[insert_location] += 1

    TPRs = []
    snr_bins_plot = []
    for i in range(nbins):
        if am_bin_total[i] != 0:
            TPRs.append(am_bin_detected[i]/am_bin_total[i])
            snr_bins_plot.append(am_bins[i]) #only adding it if nonzero total in that bin

    axs.plot(snr_bins_plot, TPRs, label="BBH", c="blue")

    plt.grid(True)
    fig.tight_layout()
    plt.savefig(f'{args.savedir}/bbh_ROC.pdf', dpi=300)
    plt.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('data', type=str,
                        help='Input signal dataset')
    parser.add_argument('data_path', type=str,
                        help='File containing the timeslides')
    parser.add_argument('save_file', type=str,
                        help='Where to save the trained model')
    parser.add_argument('savedir', type=str,
                        help='Where to save the loss plots')
    # Additional argument

    parser.add_argument('--gpu', type=str, default='0',
                        help='On which GPU to run')

    args = parser.parse_args()

    main(args)