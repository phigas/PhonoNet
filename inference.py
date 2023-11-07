import os 
import numpy as np
from train_model import load_model
from preprocess import load_datasets
import torch
from torch import nn


if __name__=='__main__':
    # define parameters
    GEOMETRY_NAME           = 'circular_hole'
    MODEL_NAME              = 'first_test'
    PHONON_PATH_LEN         = 30
    PHONON_GROUP_NUM        = 5
    
    
    # load checkpoint
    checkpoint_folder = os.path.join('checkpoints', GEOMETRY_NAME, MODEL_NAME)
    checkpoint = load_model(checkpoint_folder)
    
    # define network
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(6, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 3),
            )

        def forward(self, x):
            output = self.model(x)
            return output
    
    # choose training device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" # mac devices
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    # create model and send to device
    model = NeuralNetwork().to(device)
    print('\n=== Model structure ===')
    print(model, '\n')
    
    # compare saved model structure to loaded
    with open(os.path.join(checkpoint_folder, 'model_structure.txt'), "r") as file:
        assert file.read() == str(model), 'Make sure the model has the same structure'
    
    # set model parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # set model to evaluation mode
    model.eval()
    
    # load dataset
    _, _, test_dataloader = load_datasets(GEOMETRY_NAME)
    
    # initialize results tensor
    phonon_paths = torch.zeros((PHONON_GROUP_NUM*test_dataloader.batch_size, (PHONON_PATH_LEN+2)*3))
    
    # use train dataset as start points for phonon paths (should generate the start points on the hot side in the future)
    for i, batch in enumerate(test_dataloader):
        if i >= PHONON_GROUP_NUM: break
        
        # get shallow copy of slice for this loop
        phonon_paths_slice = phonon_paths[i*test_dataloader.batch_size:(i+1)*test_dataloader.batch_size, :]
        
        # populate matrix with using dataset
        data, _ = batch
        phonon_paths_slice[:, :6] = data
        
        # generate new paths
        for u in range(PHONON_PATH_LEN):
            inputs = phonon_paths_slice[:, u*3:u*3+6]
            inputs = inputs.to(device, dtype=torch.float32) # float32 necessary on mps devices
            outputs = model(inputs)
            phonon_paths_slice[:, u*3+6:u*3+9] = outputs        
        
phonon_paths = phonon_paths.detach().numpy()
np.savetxt(os.path.join(checkpoint_folder, 'generated_paths.csv'), phonon_paths, delimiter=',')
    