from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
import matplotlib.pyplot as plt

CHANNEL = 'DCS-CALIB_STRAIN_CLEAN_C01'

# START, END = 1267617687.94-100, 1267617687.94+100
START, END = 1249200400-100.5, 1249200400+100.5
S_RATE = 4096
LOW = 30
HIGH = 1500

data_l1_open = TimeSeries.fetch_open_data('L1', START, END).resample(S_RATE).bandpass(LOW, HIGH).whiten()
data_h1_open = TimeSeries.fetch_open_data('H1', START, END).resample(S_RATE).bandpass(LOW, HIGH).whiten()

data_l1 = TimeSeries.get(f'L1:{CHANNEL}', START, END).resample(S_RATE).bandpass(LOW, HIGH).whiten()
data_h1 = TimeSeries.get(f'H1:{CHANNEL}', START, END).resample(S_RATE).bandpass(LOW, HIGH).whiten()

plt.plot(data_h1_open[int(100.3*S_RATE):int(100.8*S_RATE)])
    # , data_l1_open[int(100.3*S_RATE):int(100.8*S_RATE)]],
    # [data_h1[int(100.3*S_RATE):int(100.8*S_RATE)], data_l1[int(100.3*S_RATE):int(100.8*S_RATE)]],
    # figsize=(12, 4.8),
    # separate=True,
    # sharex=True)
plt.show()

# import numpy as np
# import gwpy
# import matplotlib.pyplot as plt
# from gwpy.timeseries import TimeSeries

# CHANNEL = 'DCS-CALIB_STRAIN_CLEAN_C01'
# FRAME_TYPE = 'HOFT_C01'
# STATE_FLAG = 'DCS-ANALYSIS_READY_C01:1'

# def plot_open(start_point, end_point):
#     # Load LIGO data

#     strainL1 = TimeSeries.fetch_open_data(f'L1', start_point, end_point)
#     strainH1 = TimeSeries.fetch_open_data(f'H1', start_point, end_point)

#     # Whiten, bandpass, and resample
#     sample_rate = 4096
#     bandpass_low = 30
#     bandpass_high = 1500
#     strainL1 = strainL1.resample(sample_rate)
#     strainL1 = strainL1.bandpass(bandpass_low, bandpass_high).whiten()
#     strainH1 = strainH1.resample(sample_rate)
#     strainH1 = strainH1.bandpass(bandpass_low, bandpass_high).whiten()

#     plt.figure()
#     plt.plot(strainL1[int(100.3*sample_rate):int(100.8*sample_rate)])
#     plt.plot(strainH1[int(100.3*sample_rate):int(100.8*sample_rate)])
#     plt.show()


# def plot(start_point, end_point):
#     # Load LIGO data

#     strainL1 = TimeSeries.get(f'L1:{CHANNEL}', start_point, end_point)
#     strainH1 = TimeSeries.get(f'H1:{CHANNEL}', start_point, end_point)
#     # Whiten, bandpass, and resample
#     sample_rate = 4096
#     bandpass_low = 30
#     bandpass_high = 1500
#     strainL1 = strainL1.resample(sample_rate)
#     strainL1 = strainL1.bandpass(bandpass_low, bandpass_high).whiten()
#     strainH1 = strainH1.resample(sample_rate)
#     strainH1 = strainH1.bandpass(bandpass_low, bandpass_high).whiten()

#     plt.figure()
#     plt.plot(strainL1[int(100.3*sample_rate):int(100.8*sample_rate)])
#     plt.plot(strainH1[int(100.3*sample_rate):int(100.8*sample_rate)])
#     plt.show()


# detection_point = 1267617687.94
# plot(detection_point - 100.5, detection_point+100.5)
# plot_open(detection_point - 100.5, detection_point+100.5)