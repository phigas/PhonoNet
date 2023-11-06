# PhonoNet

The goal of this code is to train a neural network that can generate phonon paths faster than they can be generated with freepaths.

## How to use

- Execute your freepaths config so that the Results are put as a subfolder into the Results folder.
  - Make sure that the `OUTPUT_TRAJECTORIES_OF_FIRST` is set to the ponon number or the ponon paths won't be saved.
- Run `preprocess.py` and make sure to set the paremeters you want in the file. This will generate the datasets in the Datasets folder.
  - If you want to change the batch size later you will have to regenerate the datasets.
- Run the `train_model.py` and make sure to set the parameters you want. Multiple NN structures can be tested on one dataset by changing the `MODEL_NAME` parameter. The script will save checkpoints and tensorboard log data in a checkpoints folder.
  - If the script is run with the same `MODEL_NAME` it will five the options to continue training from the last checkpoint. This way the learning rate can be changed.
- TODO: inference script

## TODO

- Load and inference functions
- Visualize results function
- can the network use the two points that are given to "unfairly" get the next one??
- Maybe use dataset_name so that multiple datasets can be built on one set of data?
