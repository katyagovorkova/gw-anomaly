# Version of the code with which the data was generated
VERSION = 'v2'

# data generation
IFOS = ['H1', 'L1']
STRAIN_START = 1238166018
STRAIN_STOP = 1238170289
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
DO_SMOOTHING = True
N_SMOOTHING_KERNEL = 10
DATA_EVAL_MAX_BATCH = 100

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
SEGMENT_OVERLAP = 5
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

# training
TEST_SPLIT = 0.9
BOTTLENECK = {
    'bbh': 4,
    'sglf': 8,
    'sghf': 8,
    'glitches': 6,
    'background': 8}
MODEL = {
    'bbh': 'lstm',
    'sglf': 'lstm',
    'sghf': 'lstm',
    'glitches': 'dense',
    'background': 'dense'}
FACTOR = 2
EPOCHS = 200
BATCH_SIZE = 512
LOSS = 'MAE'
OPTIMIZER = 'Adam'
VALIDATION_SPLIT = 0.15
TRAINING_VERBOSE = True
CLASS_ORDER = ['background', 'bbh', 'glitches', 'sglf', 'sghf']
LIMIT_TRAINING_DATA = None
CURRICULUM_SNRS = [256, 128, 64, 32, 16]

# pearson calculation
MAX_SHIFT = 10e-3
SHIFT_STEP = 2

# timeslides
GW_EVENT_CLEANING_WINDOW = 5
TIMESLIDE_STEP = 0.5
TIMESLIDE_TOTAL_DURATION = 100 * 24 * 3600 # run on 4 different GPUs, so in total 400 * 24 * 3600
FM_TIMESLIDE_TOTAL_DURATION = 0.1 * 30 * 24 * 3600

# GPU
GPU_NAME = 'cuda:1'

# evolutionary search
INIT_SIGMA = 0.5
POPULATION_SIZE = 100
N_ELITE = 10
NOISE = 0.01

# linear SVM
SVM_LR = 0.01
N_SVM_EPOCHS = 5000

# plotting
SPEED = True
RECREATION_LIMIT = 50
RECREATION_SAMPLES_PER_PLOT = 1
RECREATION_WIDTH = 17
RECREATION_HEIGHT_PER_SAMPLE = 7
IFO_LABELS = ['Hanford', 'Livingston']
SNR_VS_FAR_BAR = 5
SNR_VS_FAR_HORIZONTAL_LINES = [3600, 24 * 3600,
                               7 * 24 * 3600, 30 * 24 * 3600, 365 * 24 * 3600]
SNR_VS_FAR_HL_LABELS = ['hour', 'day', 'week', 'month', 'year']

# varying SNR injection
N_VARYING_SNR_INJECTIONS = 5000
VARYING_SNR_DISTRIBUTION = 'uniform'
VARYING_SNR_LOW = 5
VARYING_SNR_HIGH = 50
VARYING_SNR_SEGMENT_INJECTION_LENGTH = 5

# false alarm rate calculation
HISTOGRAM_BIN_DIVISION = 0.001
HISTOGRAM_BIN_MIN = 50.

# supernova injection
SNR_SN_LOW = VARYING_SNR_LOW
SNR_SN_HIGH = VARYING_SNR_HIGH

RETURN_INDIV_LOSSES = True
SCALE = 3
