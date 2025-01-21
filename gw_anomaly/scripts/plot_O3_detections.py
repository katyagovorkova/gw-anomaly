import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Define the directory where saved data is stored
DATA_DIR = "paperO3a"

# Ensure the directory exists
if not os.path.exists(DATA_DIR):
    raise ValueError(f"Directory '{DATA_DIR}' does not exist.")

# List of specific .npz files to process
npz_files = [
    "1241104276.737_0_-1.86.npz",
    "1251009283.725_0_-4.77.npz",
    "1263013387.044_0_-7.33.npz",
    "1243305692.933_0_-4.50.npz"
]

# Define custom colormap for Q-transforms
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#1f77b4", "#f4a3c1", "#ffd700"], N=256)
vmin, vcenter, vmax = 0, 12.5, 25  # Pink at 12.5, scale from 0 to 25
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

# Define labels and colors
labels = [
    'Background', 'Background', 'BBH', 'BBH', 'Glitch', 'Glitch',
    'SG (64-512 Hz)', 'SG (64-512 Hz)', 'SG (512-1024 Hz)', 'SG (512-1024 Hz)', 'Frequency correlation'
]
colors = [
    "#f4a3c1", "#ffd700", "#2a9d8f", "#708090", "#00bfff",
    "#cd5c5c", "#006400", "#daa520", "#ff6347", "black"
]

# Process each saved file
for npz_file in npz_files:
    file_path = os.path.join(DATA_DIR, npz_file)

    if not os.path.exists(file_path):
        print(f"Warning: {npz_file} not found in {DATA_DIR}. Skipping.")
        continue

    data = np.load(file_path, allow_pickle=True)['arr_0'].item()

    # Extract data
    strain_ts, strain_chunks = data["strain"]
    gwak_time, gwak_values = data["gwak_values"]
    t_H, f_H, H_qtransform = data["H_qtransform"]
    t_L, f_L, L_qtransform = data["L_qtransform"]

    # Create figure and subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    # Hanford Q-transform
    im_H = axs[0].pcolormesh(
        t_H, f_H, H_qtransform, cmap=custom_cmap, norm=norm, shading="auto"
    )
    axs[0].set_yscale("log")
    axs[0].set_ylabel("Frequency (Hz)", fontsize=14)
    axs[0].tick_params(axis='x', which='both', bottom=False, top=False)
    axs[0].set_xticklabels([])

    # Livingston Q-transform
    im_L = axs[1].pcolormesh(
        t_L, f_L, L_qtransform, cmap=custom_cmap, norm=norm, shading="auto"
    )
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Frequency (Hz)", fontsize=14)
    axs[1].tick_params(axis='x', which='both', bottom=False, top=False)
    axs[1].set_xticklabels([])

    # Strain Time Series Plot
    axs[2].plot(strain_ts, strain_chunks[0], label='Hanford', alpha=0.8, c="#6c5b7b")
    axs[2].plot(strain_ts, strain_chunks[1], label='Livingston', alpha=0.8, c="#f29e4c")
    axs[2].set_ylabel('Strain', fontsize=14)
    axs[2].legend()
    axs[2].tick_params(axis='x', which='both', bottom=False, top=False)
    axs[2].set_xticklabels([])

    # GWAK Evaluation Plot
    axs[3].plot(gwak_time, gwak_values, label="Final Metric", c="black")
    axs[3].set_xlabel("Time (ms)", fontsize=14)
    axs[3].set_ylabel("Final Metric Contributions", fontsize=14)
    axs[3].legend()

    # Add shared colorbar
    cbar = fig.colorbar(im_H, ax=axs[0], location="top", pad=0.05, aspect=30)
    cbar.set_label("Spectral Power", fontsize=12)

    # Save figure
    save_path = os.path.join(DATA_DIR, f"{os.path.splitext(npz_file)[0]}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {save_path}")