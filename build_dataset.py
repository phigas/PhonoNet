import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


def build_dataset(geometry_name, train_split, batch_size):

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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    GEOMETRY_NAME   = 'circular_hole'
    TRAIN_SPLIT     = 0.7
    BATCH_SIZE      = 4
    
    train_dataloader, test_dataloader = build_dataset(GEOMETRY_NAME, TRAIN_SPLIT, BATCH_SIZE)
    
    a = next(iter(train_dataloader))
    
    print(a[0].shape, a[1].shape)
    print(a)