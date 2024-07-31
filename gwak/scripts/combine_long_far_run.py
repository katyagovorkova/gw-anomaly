import numpy as np
import os


def main(path, savedir = "./output"):
    total_livetime = 0
    try:
        os.makedirs(f"{savedir}/midway/")
        
    except FileExistsError:
        None

    all_scaled_evals = []
    all_final_scores = []
    for folder in os.listdir(path):
        path_ = f"{path}/{folder}"
        for folder_ in os.listdir(path_):
            path__ = f"{path_}/{folder_}"
            #if f"timeslide_data_{folder_}.npy" in os.listdir(f"{savedir}/midway/"):
            #   continue

            # load the tracker file
            #print(path__)
            try:
                livetime_loaded = np.load(f"{path__}/livetime_tracker.npy")[0]
            except ValueError:
                #print("pickle error")
                continue

            total_livetime += livetime_loaded
            N_events = len(os.listdir(path__))-1
            scaled_evals = np.zeros((N_events, 16))
            final_score = np.zeros(N_events)
            timeslide_data = np.zeros((N_events, 2, 2049))
            i = 0
            cut = 0
            print(folder_, N_events)
            print()
            for file in os.listdir(path__):
                #print(file)
                if file != "livetime_tracker.npy":
                    try:
                        data = np.load(f"{path__}/{file}")
                    except ValueError:
                        continue

                    if 'timeslide_data' in list(data.keys()):
                        timeslide_datum = data['timeslide_data'][0]  
                        if timeslide_datum.shape == (2, 2049):
                            timeslide_data[i, :, :] = timeslide_datum     
                            scaled_evals[i, :] = data['final_scaled_evals'][0]
                            final_score[i] = data['metric_score'][0]
                            i += 1
                        else:
                            cut += 1
                    else:
                        cut += 1
                    
                    
            np.save(f"{savedir}/midway/scaled_evals_{folder_}.npy", scaled_evals[:-cut])
            np.save(f"{savedir}/midway/final_score_{folder_}.npy", final_score[:-cut])
            np.save(f"{savedir}/midway/timeslide_data_{folder_}.npy", timeslide_data[:-cut])

            all_scaled_evals.append(scaled_evals)
            all_final_scores.append(final_score)

            print(f"Total livetime: {total_livetime/3.15e7 :.2f} years")

    all_scaled_evals = np.concatenate(all_scaled_evals, axis=0)
    all_final_scores = np.concatenate(all_final_scores, axis=0)
    print("scaled evals", all_scaled_evals.shape)
    print("scores", all_final_scores.shape)

    np.save(f"{savedir}/scaled_evals.npy", all_scaled_evals)
    np.save(f"{savedir}/final_scores.npy", all_final_scores)
    print()
    print(total_livetime/3.15e7)



if __name__ == "__main__":
    main('/home/ryan.raikman/far_comp/')
