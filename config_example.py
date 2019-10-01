# =============== Basic Configurations ===========
PYRAMID_W = 512
PYRAMID_H = 512


# =============== Train Configurations ===========
DATA_DIR = ''
CHECKPOINT_DIR = ''
LOG_DIR = ''
TRAIN_SET = ['{:04d}'.format(i) for i in range(899)]
EPOCH = 10
BATCH_SIZE = 64
CROP_W = 1024
CROP_H = 512
LEARNING_RATE = 1e-3
BETAS = (0.9, 0.999)
L2_WEIGHT_DECAY = [0.01, 0.001, 0.0001, 0]
EPS = 1e-8