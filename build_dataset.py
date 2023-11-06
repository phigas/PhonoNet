import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


def build_dataset(geometry_name, train_val_test_split, batch_size, num_workers):

    assert sum(train_val_test_split) == 1, 'train, test and val splits need to add up to 1'
    
    # load the data as numpy
    data = np.load(os.path.join('Datasets', f'{geometry_name}.npy'))

    # split the data into input and output
    input_data = data[:,:-3]
    output_data = data[:,-3:]

    # convert to tensor
    input_data = torch.from_numpy(input_data)
    output_data = torch.from_numpy(output_data)

    # build Dataset
    dataset = TensorDataset(input_data, output_data)

    # split dataset into train and test
    train_dataset, val_dataset, test_dataset = random_split(dataset, train_val_test_split)

    # Wrap in dataloader for batching, shuffling, multiprocessing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    GEOMETRY_NAME           = 'circular_hole'
    TRAIN_VAL_TEST_SPLIT    = (0.6, 0.2, 0.2)
    BATCH_SIZE              = 4
    NUM_WORKERS             = 0
    
    train_dataloader, val_dataloader, test_dataloader = build_dataset(GEOMETRY_NAME, TRAIN_VAL_TEST_SPLIT, BATCH_SIZE, NUM_WORKERS)
    
    a = next(iter(train_dataloader))
    
    print(a[0].shape, a[1].shape)
    print(a)
    
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))