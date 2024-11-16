from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot

CHANNEL = 'DCS-CALIB_STRAIN_CLEAN_C01'

START, END = 1267617687.94-100, 1267617687.94+100
S_RATE = 4096
LOW = 30
HIGH = 1500

data_l1_open = TimeSeries.fetch_open_data('L1', START, END).resample(S_RATE).whiten().bandpass(LOW, HIGH)
data_h1_open = TimeSeries.fetch_open_data('H1', START, END).resample(S_RATE).whiten().bandpass(LOW, HIGH)

data_l1 = TimeSeries.get(f'L1:{CHANNEL}', START, END).resample(S_RATE).whiten().bandpass(LOW, HIGH)
data_h1 = TimeSeries.get(f'H1:{CHANNEL}', START, END).resample(S_RATE).whiten().bandpass(LOW, HIGH)

pl_start, pl_end = int(len(data_h1_open)/2)-500, int(len(data_h1_open)/2)+500
plot = Plot(
    [data_h1_open, data_l1_open],
    [data_h1, data_l1],
    figsize=(12, 4.8),
    separate=True,
    sharex=True)
plot.show()

# import numpy as np
# import gwpy
# import matplotlib.pyplot as plt
# from gwpy.timeseries import TimeSeries

# CHANNEL = 'DCS-CALIB_STRAIN_CLEAN_C01'
# FRAME_TYPE = 'HOFT_C01'
# STATE_FLAG = 'DCS-ANALYSIS_READY_C01:1'

# def plot_qscans(start_point, end_point):
#     # Load LIGO data

#     strainL1 = TimeSeries.fetch_open_data(f'L1', start_point, end_point)
#     strainH1 = TimeSeries.fetch_open_data(f'H1', start_point, end_point)

#     # strainL1 = TimeSeries.get(f'L1:{CHANNEL}', start_point, end_point)
#     # strainH1 = TimeSeries.get(f'H1:{CHANNEL}', start_point, end_point)
#     # Whiten, bandpass, and resample
#     sample_rate = 4096
#     bandpass_low = 30
#     bandpass_high = 1500
#     strainL1 = strainL1.resample(sample_rate)
#     strainL1 = strainL1.bandpass(bandpass_low, bandpass_high).whiten()
#     strainH1 = strainH1.resample(sample_rate)
#     strainH1 = strainH1.bandpass(bandpass_low, bandpass_high).whiten()

#     np.save('strainL1.npy', strainL1[int(100.3*sample_rate):int(100.8*sample_rate)].value)
#     np.save('strainH1.npy', strainH1[int(100.3*sample_rate):int(100.8*sample_rate)].value)

#     plt.figure()
#     plt.plot(strainL1[int(100.3*sample_rate):int(100.8*sample_rate)])
#     plt.plot(strainH1[int(100.3*sample_rate):int(100.8*sample_rate)])
#     plt.show()

#     # H_hq = strainH1.q_transform(outseg=(start_point+100, end_point-100), whiten=False)
#     # L_hq = strainL1.q_transform(outseg=(start_point+100, end_point-100), whiten=False)

#     # vmax = 25.0
#     # vmin = 0

#     # plot = H_hq.plot(figsize=[8, 4], vmax=vmax, vmin=vmin)
#     # ax = plot.gca()
#     # ax.set_xscale('seconds')
#     # ax.set_yscale('log')
#     # ax.set_ylim(20, 500)
#     # ax.set_ylabel('Frequency [Hz]')
#     # ax.grid(True, axis='y', which='both')
#     # ax.colorbar(cmap='viridis', label='Normalized energy')
#     # plot.show()

#     # plot = L_hq.plot(figsize=[8, 4], vmax=vmax, vmin=vmin)
#     # ax = plot.gca()
#     # ax.set_xscale('seconds')
#     # ax.set_yscale('log')
#     # ax.set_ylim(20, 500)
#     # ax.set_ylabel('Frequency [Hz]')
#     # ax.grid(True, axis='y', which='both')
#     # ax.colorbar(cmap='viridis', label='Normalized energy')
#     # plot.show()


# detection_point = 1267617687.94
# plot_qscans(detection_point - 100.5, detection_point+100.5)
