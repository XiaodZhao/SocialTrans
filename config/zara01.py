# model
PERCEP_RADIUS = 2       # perception radius, neighborhood radius
OB_HORIZON = 8      # number of observation frames
PRED_HORIZON = 8   # number of prediction frames
# group name of inclusive agents; leave empty to include all agents
# non-inclusive agents will appear as neighbors only
INCLUSIVE_GROUPS = []

# training
LEARNING_RATE = 3e-4 
BATCH_SIZE = 32
EPOCHS = 600          # total number of epochs for training
EPOCH_BATCHES = 100   # number of batches per epoch, None for data_length//batch_size
TEST_SINCE = 200      # the epoch after which performing testing during training

# testing
PRED_SAMPLES = 20   # best of N samples

# evaluation
WORLD_SCALE = 1
