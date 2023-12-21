# Version of the code with which the data was generated
PERIOD = 'O3a' # or O3b
VERSION = PERIOD + 'v2' # _only_correlation
STRAIN_START = 1238166018 # for O3b 1256663958 1238166018
STRAIN_STOP = 1238170289 # for O3b 1256673192 1238170289
# GPU
GPU_NAME = 'cuda:0'
# data generation
IFOS = ['H1', 'L1']
SAMPLE_RATE = 4096
BANDPASS_LOW = 30
BANDPASS_HIGH = 1500
GLITCH_SNR_BAR = 10
N_TRAIN_INJECTIONS = 20000
N_TEST_INJECTIONS = 500
N_FM_INJECTIONS = 500
LOADED_DATA_SAMPLE_RATE = 16384
DATA_SEGMENT_LOAD_START = 0
DATA_SEGMENT_LOAD_STOP = 3600
EDGE_INJECT_SPACING = 1.2
TRAIN_INJECTION_SEGMENT_LENGTH = 4
FM_INJECTION_SEGMENT_LENGTH = 5
FM_INJECTION_SNR = 20

# data sampling arguments
BBH_WINDOW_LEFT = -0.08
BBH_WINDOW_RIGHT = 0.01
BBH_AMPLITUDE_BAR = 5
BBH_N_SAMPLES = 5
SG_WINDOW_LEFT = -0.05
SG_WINDOW_RIGHT = 0.05
SG_AMPLITUDE_BAR = 5
SG_N_SAMPLES = 5
BKG_N_SAMPLES = 5
NUM_IFOS = 2
SEG_NUM_TIMESTEPS = 200
SEGMENT_OVERLAP = 50
SIGNIFICANCE_NORMALIZATION_DURATION = 10
GLITCH_WINDOW_LEFT = -0.05
GLITCH_WINDOW_RIGHT = 0.05
GLITCH_N_SAMPLES = 20
GLITCH_AMPLITUDE_BAR = 5

# Glitch 'generation'
SNR_THRESH = 5
Q_MIN = 3.3166
Q_MAX = 108
F_MIN = 32
CLUSTER_DT = 0.5
CHUNK_DURATION = 124
SEGMENT_DURATION = 64
OVERLAP = 4
MISTMATCH_MAX = 0.2
WINDOW = 2
CHANNEL = 'DCS-CALIB_STRAIN_CLEAN_C01'
FRAME_TYPE = 'HOFT_C01'
GLITCH_SAMPLE_RATE = 1024
STATE_FLAG = 'DCS-ANALYSIS_READY_C01:1'

# timeslides
GW_EVENT_CLEANING_WINDOW = 5
TIMESLIDE_STEP = 0.5
TIMESLIDE_TOTAL_DURATION = int(1.25 * 365 * 24 * 3600 / 4) # run on 4 different GPUs, so in total 400 * 24 * 3600
FM_TIMESLIDE_TOTAL_DURATION = 0.1 * 30 * 24 * 3600
TIMESLIDES_START = 1238166018 # Ryan = 1248652818; Eric = 1243382418; Katya = 1238166018
TIMESLIDES_STOP =  1243382418 # Ryan = 1253977218; Eric = 1248652818; Katya = 1243382418
