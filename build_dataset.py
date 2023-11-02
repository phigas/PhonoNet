import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


def build_dataset(geometry_name, train_split, batch_size, num_workers):

    # load the data as numpy
    data = np.load(os.path.join('Datasets', f'{geometry_name}.npy'))

    # split the data into input and output
    input_data = data[:,:-1,:]
    output_data = data[:,-1:,:]

    # convert to tensor
    input_data = torch.from_numpy(input_data)
    output_data = torch.from_numpy(output_data)

    # build Dataset
    dataset = TensorDataset(input_data, output_data)

    # split dataset into train and test
    train_dataset, test_dataset = random_split(dataset, [train_split, 1-train_split])

    # Wrap in dataloader for batching, shuffling, multiprocessing
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    GEOMETRY_NAME   = 'circular_hole'
    TRAIN_SPLIT     = 0.7
    BATCH_SIZE      = 4
    NUM_WORKERS     = 0
    
    train_dataloader, test_dataloader = build_dataset(GEOMETRY_NAME, TRAIN_SPLIT, BATCH_SIZE, NUM_WORKERS)
    
    a = next(iter(train_dataloader))
    
    print(a[0].shape, a[1].shape)
    print(a)