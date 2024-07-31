import numpy as np
import matplotlib.pyplot as plt


# this was given to me by eric from the super supercomputer, so match this with wherever it comes out of
# TODO: reconfigure
data = np.load("/home/eric.moreno/combined_heuristics_data_11k.npy")
data_cut = data[np.logical_and(data[:, -1]>-50, data[:, -1]<-3)] # excize "crazy" events - the suspicion was that these were coming from some weird behavior in evaluate_timeslides
                                                                # as well as anything above metric score -3
resampled_heuristics = data_cut[np.random.permutation(len(data_cut))]

# NOTE: it would be optimal if we ran timeslides on the 1000 years first, used it as train, and passed the rest into REAL
# this way, if we want to save plots of things that were detected in timeslides (with the GOAL of showing which events are more significant than ours)
# then we can already aplpy heuristics and there will be many less plots to deal with
train, REAL = resampled_heuristics[:len(reduced_heuristics)//10], resampled_heuristics[len(reduced_heuristics)//10:]
# REAL is the data used to finally compute the false alarm rates

# TODO: reconfigure
heurustics_train_path = "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/O3av2/evaluated/heuristic/real_bkg_data.npy"
np.save(heuristics_train_path, train)

# TODO: reconfigure
far_data_path = "/home/ryan.raikman/s22/forks/katya/gw-anomaly/output/O3av2/evaluated/eval_heur.npy"
np.save(far_data_path, REAL)



# TODO: for the signal data for training the heuristic model, use the function in evaluate_data.py
# it's on lns 223-295, there's a flag 
# data_eval_use_heuristic = False on line 223 that should be flipped
# tldr, run this with the flag = True