# Version of the code with which the data was generated
PERIOD = 'O3a' # or O3b
VERSION = PERIOD + 'v2' # _only_correlation
# if you use only O3a/O3b, you need to remove the other period below:
STRAIN_START_STOP = {
    'O3a': [1238166018, 1238170289],
    # 'O3b': [1256663958, 1256673192]
    }


# If you want to use specific version of data or models, specify the path here
# and in the Snakefile turn on a flag that says use_trained_models
DATA_LOCATION = f'/home/katya.govorkova/gwak-paper-final-models/data'
MODELS_LOCATION = f'/home/katya.govorkova/gwak-paper-final-models/trained/models/'
FM_LOCATION = f'/home/katya.govorkova/gwak-paper-final-models/trained/'

# GPU
GPU_NAME = 'cuda:1'

# Data generation
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
DO_SMOOTHING = True
SMOOTHING_KERNEL = 50
SMOOTHING_KERNEL_SIZES = [1, 10, 50, 100]
DATA_EVAL_MAX_BATCH = 100

# Data sampling arguments
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

# Training
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

# Pearson calculation
MAX_SHIFT = 10e-3
SHIFT_STEP = 2
PEARSON_FLAG = False

"""
    Factors to keep for the FM
    0 - background AE (L_O * L_R)
    1 - background AE (H_O * H_R)
    2 - background AE (L_O * H_O)
    3 - background AE (L_R * H_R)
    4 - BBH AE (L_O * L_R)
    5 - BBH AE (H_O * H_R)
    6 - BBH AE (L_O * H_O)
    7 - BBH AE (L_R * H_R)
    8 - Glitches AE (L_O * L_R)
    9 - Glitches AE (H_O * H_R)
    10 - Glitches AE (L_O * H_O)
    11 - Glitches AE (L_R * H_R)
    12 - SGLF AE (L_O * L_R)
    13 - SGLF AE (H_O * H_R)
    14 - SGLF AE (L_O * H_O)
    15 - SGLF AE (L_R * H_R)
    16 - SGHF AE (L_O * L_R)
    17 - SGHF AE (H_O * H_R)
    18 - SGHF AE (L_O * H_O)
    19 - SGHF AE (L_R * H_R)
    20 - Pearson
"""
# Baseline
FACTORS_NOT_USED_FOR_FM = [3,7,11,15,19]

# Timeslides
GW_EVENT_CLEANING_WINDOW = 5
TIMESLIDE_STEP = 0.5
TIMESLIDE_TOTAL_DURATION = int(1.1 * 365 * 24 * 3600 / 4)*4 # run on 4 different GPUs, so in total 400 * 24 * 3600
FM_TIMESLIDE_TOTAL_DURATION = 0.1 * 30 * 24 * 3600
TIMESLIDES_START = 1249093442#1238166018
TIMESLIDES_STOP = 1249101026#1243382418

# Linear SVM
SVM_LR = 0.01
N_SVM_EPOCHS = 5000

# Plotting
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

# Varying SNR injection
N_VARYING_SNR_INJECTIONS = 2000
VARYING_SNR_DISTRIBUTION = 'uniform'
VARYING_SNR_LOW = 5
VARYING_SNR_HIGH = 50
VARYING_SNR_SEGMENT_INJECTION_LENGTH = 5

# False alarm rate calculation
HISTOGRAM_BIN_DIVISION = (20 - (-20))/1000
HISTOGRAM_BIN_MIN = 20.

# Supernova injection
SNR_SN_LOW = VARYING_SNR_LOW
SNR_SN_HIGH = VARYING_SNR_HIGH
N_BURST_INJ = 500

RETURN_INDIV_LOSSES = True
SCALE = 3

# Supervised
SUPERVISED_BKG_TIMESLIDE_LEN = 24*3600
SUPERVISED_N_BKG_SAMPLE = 1000
SUPERVISED_REDUCE_N_BKG = 85000
SUPERVISED_BATCH_SIZE = 1280
SUPERVISED_EPOCHS = 100
SUPERVISED_VALIDATION_SPLIT = 0.15
SUPERVISED_FAR_TIMESLIDE_LEN = 24*3600
SUPERVISED_SMOOTHING_KERNEL = 50

# Heuristics
DATA_EVAL_USE_HEURISTIC = True
