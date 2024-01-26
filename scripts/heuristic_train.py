import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import torch.optim as optim
#from config import GPU_NAME
GPU_NAME = "cuda:2"
device = torch.device(GPU_NAME)
class HeuristicModel(nn.Module):
    def __init__(self):
        super(HeuristicModel, self).__init__()

        self.layer1 = nn.Linear(6, 4)
        self.layer2 = nn.Linear(4, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return self.activation(x)
    
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

data_path = "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/O3av2/evaluated/heuristic/"

def extract(gwak_values):
    result = np.zeros((gwak_values.shape[0], 3))
    for i, pair in enumerate([[3, 4], [9, 10], [12, 13]]):
        a, b = pair
        ratio_a = (np.abs(gwak_values[:, a]) + 2) / (np.abs(gwak_values[:, b]) + 2)
        ratio_b = (np.abs(gwak_values[:, a]) + 2) / (np.abs(gwak_values[:, a]) + 2)

        ratio = np.maximum(ratio_a, ratio_b)
        result[:, i] = ratio

    return result

bkg_data = [[],[]]
signal_data = {}
for elem in os.listdir(data_path):
    class_name = elem.split("_")[0]
    if class_name != "bkg": 
        class_name = elem.split("_")[0]
        signal_data[class_name] = [[],[]]

bkg_metric_score = None
for elem in os.listdir(data_path):
    data = np.load(f"{data_path}/{elem}")
    class_name = elem.split("_")[0]
    tag = elem.split("_")[1]

    if class_name == "bkg": 
        if tag == "gwak":
            bkg_data[1] = extract(data)
        elif tag == "metric":
            bkg_metric_score = data
        else:
            bkg_data[0] = data

    else:
        if tag == "gwak":
            signal_data[class_name][1] = extract(data)
        else:
            signal_data[class_name][0] = data

sig_data = []
for key in signal_data.keys():
    sig_data.append(np.concatenate(signal_data[key],axis=1))
sig_data = np.concatenate(sig_data, axis=0)
bkg_data = np.concatenate(bkg_data,axis=1)

def sig_loss_function(evals, scale=40):
    return torch.mean(torch.sigmoid(scale * (evals-0.5)))

def bkg_loss_function(evals, fm_weights, scale=40):
    #print( (1 - torch.sigmoid(scale * (evals-0.5))).shape, fm_weights.shape)
    return torch.mean( (1 - torch.sigmoid(scale * (evals-0.5))) *fm_weights)

#def bkg_loss_function(evals, fm_weights, scale=40):
#    #print( (1 - torch.sigmoid(scale * (evals-0.5))).shape, fm_weights.shape)
#    return torch.mean( (1 - torch.sigmoid(scale * (evals-0.5))))# *fm_weights)

# training

p1 = np.random.permutation(len(sig_data))
p2 = np.random.permutation(len(bkg_data))

sig_data = torch.from_numpy(sig_data[p1]).float().to(device)
bkg_data = torch.from_numpy(bkg_data[p2]).float().to(device)
mult = np.mean(bkg_metric_score)
orig = np.copy(bkg_metric_score[p2])
fm_vals = torch.from_numpy(bkg_metric_score[p2] / np.mean(bkg_metric_score)).float().to(device)

sig_val_cut = int( len(sig_data) * 0.8)
bkg_val_cut = int( len(bkg_data) * 0.8)
sig_data_val = sig_data[sig_val_cut:]
bkg_data_val = bkg_data[bkg_val_cut:]
sig_data = sig_data[:sig_val_cut]
bkg_data = bkg_data[:bkg_val_cut]
print("train", sig_data.shape, bkg_data.shape, "val", sig_data_val.shape, bkg_data_val.shape)

saved_model = True
#model = HeuristicModel().to(device)
model = BasedModel().to(device)
model_save_path = "output/plots/model.h5"
if saved_model:
    model.load_state_dict(torch.load(model_save_path))
n_epoch = 100
optimizer = optim.Adam(model.parameters())
n_batches = 50
sig_batch_size = len(sig_data)//n_batches
bkg_batch_size = len(bkg_data)//n_batches

#for curric in range(N_curric):
#    if not curric in [2, 3, 4] : continue
#    x, y = train_set[curric]["x"][:9000], train_set[curric]["y"][:9000]
training_history = {
        'train_loss': [],
        'val_loss': [],
        'curric_step': []
    }

def acc(model_pred, class_name, bar=0.5):
    if class_name == "bkg":
        return torch.sum(model_pred > bar) / len(model_pred)
    
    return torch.sum(model_pred < bar) / len(model_pred)

if 1:
    first3 = (-0.6739309049533454, 5.607862036760292, 9.394911318980343)
    gwakscore = np.array([-0.11469545, -0.23901513,  0.14832631, -2.816604,   -4.1398506,
    0.5572696,   2.7083097,   0.49473998, -3.136547,    0.16245346,
    0.27004316, -1.374692,    0.4257469,   0.76142395, -5.0846195 ])[np.newaxis, :]

    gwakscore = extract(gwakscore)[0]
    together = np.concatenate([first3, gwakscore])[np.newaxis,]
    special = torch.from_numpy(together).float().to(device)
if not saved_model:
    for epoch in range(n_epoch):
        #print(fm_vals)
        epoch_train_loss = 0
        epoch_bkg_loss = 0
        epoch_sig_loss = 0
        for batch_num in range(n_batches):
            optimizer.zero_grad()
            start = batch_num * sig_batch_size
            end = start + sig_batch_size
            sig_batch = sig_data[start:end]
            
            output = model(sig_batch)
            loss_sig = sig_loss_function(output)

            start = batch_num * bkg_batch_size
            end = start + bkg_batch_size
            bkg_batch = bkg_data[start:end]
            #optimizer.zero_grad()
            output = model(bkg_batch)
            loss_bkg = bkg_loss_function(output, fm_vals[start:end]) * 3
            
            #special_loss = sig_loss_function(model(special)) * 0.15
            epoch_train_loss += loss_sig.item()
            epoch_train_loss += loss_bkg.item()
            epoch_sig_loss += loss_sig.item()
            epoch_bkg_loss += loss_bkg.item()
            #epoch_train_loss += special_loss.item()
            #special_loss.backward()
            loss_sig.backward()
            loss_bkg.backward()
            optimizer.step()
            
        
        val_loss = bkg_loss_function(model(bkg_data_val), fm_vals[bkg_val_cut:]).item() + sig_loss_function(model(sig_data_val)).item()
        print(f"epoch {epoch} train loss {epoch_train_loss/n_batches :.4f} val loss {val_loss :.4f} ")
        print(f"epoch {epoch} sig loss {epoch_sig_loss/n_batches :.4f} bkg loss {epoch_bkg_loss/n_batches :.4f} ")

        train_bkg_acc = acc(model(bkg_data), "bkg")
        train_sig_acc = acc(model(sig_data), "sig")

        val_bkg_acc = acc(model(bkg_data_val), "bkg")
        val_sig_acc = acc(model(sig_data_val), "sig")

        print(f"train acc: bkg {train_bkg_acc:.3f} signal {train_sig_acc:.3f}, val acc: bkg {val_bkg_acc:.3f} signal {val_sig_acc:.3f} ")
        print(f"my precious", (model(special)))
        #if model(torch.from_numpy(together).float().to(device)).item() > 0.48:
        #    break
    torch.save(model.state_dict(), model_save_path)
if 0:
    print("final")
    train_bkg_acc = acc(model(bkg_data), "bkg")
    train_sig_acc = acc(model(sig_data), "sig")

    val_bkg_acc = acc(model(bkg_data_val), "bkg")
    val_sig_acc = acc(model(sig_data_val), "sig")

    print(f"train acc: bkg {train_bkg_acc:.3f} signal {train_sig_acc:.3f}, val acc: bkg {val_bkg_acc:.3f} signal {val_sig_acc:.3f} ")

def make_roc_curve(bkg_evals, sig_evals):
    points = []
    for thresh in np.linspace(0, 1.01, 100):
        points.append([(bkg_evals < thresh).sum()/len(bkg_evals),
                       (sig_evals < thresh).sum()/len(sig_evals) ])
    points2 = []
    for thresh in [0.3, 0.35, 0.4, 0.42, 0.45, 0.5, 0.52, 0.55]:
        points2.append([(bkg_evals < thresh).sum()/len(bkg_evals),
                       (sig_evals < thresh).sum()/len(sig_evals) ])
        
    points = np.array(points)
    plt.plot(points[:, 0], points[:, 1])
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    for i, elem in enumerate(points2):
        plt.scatter(elem[0], elem[1], label = "cutoff" + str([0.3, 0.35, 0.4, 0.42, 0.45, 0.5, 0.52, 0.55][i]))
    #plt.loglog()
    plt.legend()
    plt.ylim(0.85, 1.01)
    plt.grid()
    plt.xscale("log")
    plt.savefig(f"output/plots/ROC_heuristic.png", dpi=300)
    
    plt.close()

def heuristic_cut_performance_plot(model_pred, fm_vals, save_class="bkg"):
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(model_pred, bins=200)
    axs[0].set_yscale("log")
    axs[0].set_xlim(-0.05, 1.05)
    axs[0].set_xlabel("NN output")
    axs[0].set_ylabel("count")

    if save_class == "bkg":
        #print(model_pred.shape, fm_vals.shape)
        mult = 1
        bins = np.linspace(min(fm_vals*mult)-0.5, max(fm_vals*mult)+0.5, 150)[:, 0]
        #print("bins", bins)
        counts = np.zeros(bins.shape)
        post_heur_counts = np.zeros(bins.shape)
        #print((fm_vals*mult).shape); assert 0
        for i, elem in enumerate(fm_vals*mult):
            k = np.searchsorted(bins, elem)
            #print("ELEM", elem, "\n")
            #print("k", k)
            #if i > 50:
            #    assert 0
            counts[k] += 1
            if model_pred[i] < 0.5:
                post_heur_counts[k] += 1

        #print(counts)
        #print(post_heur_counts)
        axs[1].plot(bins, counts, label = "original bin counts")
        axs[1].plot(bins, post_heur_counts, label = "passed heuristics")
        axs[1].legend()
        axs[1].set_yscale("log")
        axs[1].set_xlabel("final metric score")
        axs[1].set_ylabel("count")



    fig.savefig(f"output/plots/heuristic_cut_{save_class}.png", dpi=300)
    plt.close()

#[-6.799029]

#print(together.shape)
#assert 0


make_roc_curve(model(bkg_data_val).detach().cpu().numpy(), model(sig_data_val).detach().cpu().numpy())
heuristic_cut_performance_plot(model(bkg_data_val).detach().cpu().numpy(), orig[bkg_val_cut:], "bkg")
heuristic_cut_performance_plot(model(sig_data_val).detach().cpu().numpy(), orig[bkg_val_cut:], "sig")
if 0:
    print(model.layer1.weight)
    print(model.layer1.bias)
    print(model.layer2.weight)
    print(model.layer2.bias)
if 0:
    print(model.layer1.weight)
    print(model.layer1.bias)
    print()
    print(model.layer2_1.weight)
    print(model.layer2_1.bias)
    print()
    print(model.layer2_2.weight)
    print(model.layer2_2.bias)
    print()
    print(model.layer2_3.weight)
    print(model.layer2_3.bias)
