from build_dataset import build_dataset


# define parameters
GEOMETRY_NAME   = 'circular_hole'
TRAIN_SPLIT     = 0.7
BATCH_SIZE      = 4


# get datasets
train_dataloader, test_dataloader = build_dataset(GEOMETRY_NAME, TRAIN_SPLIT, BATCH_SIZE)

