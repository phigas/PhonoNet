import numpy as np
import os


GEOMETRY_NAME   = 'circular_hole'
DIMENSION       = 3


# matrix shouls have three times as many columns as photons (x,y,z), number of lines depends on the longest phonon travel or the time limit
phonon_paths = np.loadtxt(os.path.join('Results', GEOMETRY_NAME, 'Data/Phonon paths.csv'), delimiter=',')
print('Data loaded succesfully')

# initialize list to store samples
samples = []

for phonon_nr in range(round(phonon_paths.shape[1]/DIMENSION)):
    # Extract one phonon path
    phonon_path = phonon_paths[:,phonon_nr*DIMENSION:phonon_nr*DIMENSION+DIMENSION]
    
    # Remove lines with 0s
    phonon_path = phonon_path[~np.all(phonon_path == 0, axis=1)]
    # Create coordinate triplets for training (two coordinates as input and one as output)
    stacked_path = np.stack((phonon_path[:-2], phonon_path[1:-1], phonon_path[2:]), axis=1)
    # add to list
    samples.append(stacked_path)
     
# Build dataset matrix (threedimensional matrix with dimensions: sample, coordinate number, coordinate direction)
samples = np.concatenate(samples)
print('Data reshaped')

# save the data
np.save(os.path.join('Datasets', f'{GEOMETRY_NAME}.npy'), samples, allow_pickle=False)
print('Data saved')