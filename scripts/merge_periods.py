import os
import numpy as np
import argparse


def main(args):

    input_files = args.input_files

    # if we are running only for one period, no need to do anything,
    # just copy generated data in the final folder
    if len(input_files) == 1:
        os.system(f'cp {input_files} {output}')
        return

    print(f'Files to merge: {input_files}')
    data_all = [np.load(fname) for fname in input_files]

    if "clean" in data_all[0].keys():

        merged_data = {"clean":[], "noisy":[]}
        merged_data["clean"] = np.concatenate([data_all[0]["clean"],data_all[1]["clean"]], axis=1)
        merged_data["noisy"] = np.concatenate([data_all[0]["noisy"],data_all[1]["noisy"]], axis=1)

        rng = np.random.default_rng()
        rng.shuffle(merged_data["clean"], axis=1)
        rng.shuffle(merged_data["noisy"], axis=1)

    else:

        merged_data = {"data":[]}
        merged_data["data"] = np.concatenate([data_all[0]["data"],data_all[1]["data"]], axis=1)

        rng = np.random.default_rng()
        rng.shuffle(merged_data["data"], axis=1)

    np.savez(args.output, **merged_data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('input_files', type=str, nargs='+',
                        help='Path to the files generated with different periods')

    parser.add_argument('--output', type=str,
                        help='Where to save the file with injections')

    args = parser.parse_args()
    main(args)
