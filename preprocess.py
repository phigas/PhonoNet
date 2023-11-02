import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


if __name__ == '__main__':
    GEOMETRY_NAME   = 'circular_hole'
    THICKNESS       = 150e-9
    WIDTH           = 1000e-9
    LENGTH          = 2000e-9


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
    np.save(os.path.join('Datasets', f'{GEOMETRY_NAME}.npy'), samples, allow_pickle=False)
    print('Data saved')