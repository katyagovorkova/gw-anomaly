# data generation
IFOS = ['H1', 'L1']
STRAIN_START = 1239134846
STRAIN_STOP = 1239140924
SAMPLE_RATE = 4096
GLITCH_SNR_BAR = 10
N_INJECTIONS = 500

# Glitch "generation"
SNR_THRESH = 8
GLITCH_START = 1239134846
GLITCH_STOP = 1239140924
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
BOTTLENECK = 15
FACTOR = 2
EPOCHS = 100
BATCH_SIZE = 100
LOSS = 'MAE'
OPTIMIZER = 'Adam'
VALIDATION_SPLIT = 0.15