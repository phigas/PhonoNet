import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


def get_scaler(thickness, width, length):
    # the scaler will scale each coordinate to 0..1 in function of the simulation domain
    
    # convert m to um
    thickness, width, length = [a * 1e6 for a in [thickness, width, length]]
    
    x_min, x_max = -width/2, width/2
    y_min, y_max = 0, length
    z_min, z_max = -thickness/2, thickness/2
    
    fit_array = np.zeros((2,9))
    fit_array[0] = [x_min, y_min, z_min]*3
    fit_array[1] = [x_max, y_max, z_max]*3
    
    # create and fit the scaler
    scaler = MinMaxScaler()
    scaler.fit(fit_array)
    
    return scaler
    
def scale_data(samples, thickness, width, length):
    scaler = get_scaler(thickness, width, length)
    
    samples = scaler.transform(samples)
    
    return samples
    
def unscale_data(samples, thickness, width, length):
    scaler = get_scaler(thickness, width, length)
    
    samples = scaler.inverse_transform(samples)
    
    return samples

def build_dataset(data, train_val_test_split, batch_size, num_workers):

    assert sum(train_val_test_split) == 1, 'train, test and val splits need to add up to 1'
    
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

def load_datasets(geometry_name):
    folder = os.path.join('Datasets', geometry_name)
    
    train_dataloader = torch.load(os.path.join(folder, 'train.dl'))
    val_dataloader = torch.load(os.path.join(folder, 'val.dl'))
    test_dataloader = torch.load(os.path.join(folder, 'test.dl'))
    
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == '__main__':
    GEOMETRY_NAME           = 'circular_hole'
    THICKNESS               = 150e-9
    WIDTH                   = 1000e-9
    LENGTH                  = 2000e-9
    TRAIN_VAL_TEST_SPLIT    = (0.6, 0.2, 0.2)
    BATCH_SIZE              = 16
    NUM_WORKERS             = 0
    
    
    # check if folder created
    folder = os.path.join('Datasets', GEOMETRY_NAME)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # matrix should have three times as many columns as photons (x,y,z), number of lines depends on the longest phonon travel or the time limit
    phonon_paths = np.loadtxt(os.path.join('Results', GEOMETRY_NAME, 'Data/Phonon paths.csv'), delimiter=',')
    print('Data loaded succesfully')

    # initialize list to store samples
    samples = []

    for phonon_nr in range(round(phonon_paths.shape[1]/3)):
        # Extract one phonon path
        phonon_path = phonon_paths[:,phonon_nr*3:phonon_nr*3+3]
        
        # Remove lines with 0s
        phonon_path = phonon_path[~np.all(phonon_path == 0, axis=1)]
        # Create coordinate triplets for training (two coordinates as input and one as output)
        stacked_path = np.stack((phonon_path[:-2], phonon_path[1:-1], phonon_path[2:]), axis=1)
        # add to list
        samples.append(stacked_path)
        
    # Build dataset matrix (threedimensional matrix with dimensions: sample, coordinate number, coordinate direction)
    samples = np.concatenate(samples)
    print('Data reshaped')

    # normalize the data to 0-1 (for each coordinate individually) (alternatively data standartization might give better results)
    # the data needs to be flattened so that the scaler works and for the nn later
    print(samples[0],samples.shape)
    samples = np.reshape(samples, (-1,9))
    print(samples[0],samples.shape)
    samples = scale_data(samples, THICKNESS, WIDTH, LENGTH)
    print(samples[0])

    # save the data
    np.save(os.path.join(folder, 'full_data.npy'), samples, allow_pickle=False)
    
    # generate datasets
    train_dataloader, val_dataloader, test_dataloader = build_dataset(samples, TRAIN_VAL_TEST_SPLIT, BATCH_SIZE, NUM_WORKERS)

    # save datasets
    torch.save(train_dataloader, os.path.join(folder, 'train.dl'))
    torch.save(val_dataloader, os.path.join(folder, 'val.dl'))
    torch.save(test_dataloader, os.path.join(folder, 'test.dl'))

    # save information
    with open(os.path.join(folder, 'info.txt'), "w") as file:
        file.write(f'GEOMETRY_NAME: {GEOMETRY_NAME}\nTHICKNESS: {THICKNESS}\nWIDTH: {WIDTH}\nLENGTH: {LENGTH}\nTRAIN_VAL_TEST_SPLIT: {TRAIN_VAL_TEST_SPLIT}\nBATCH_SIZE: {BATCH_SIZE}\nNUM_WORKERS: {NUM_WORKERS}\nSAMPLES: {samples.shape}\nTRAIN_BATCHES: {len(train_dataloader)}')
        
    print('Data saved')
    
    # some diagnostics
    a = next(iter(train_dataloader))
    
    print(a[0].shape, a[1].shape)
    print(a)
    
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))