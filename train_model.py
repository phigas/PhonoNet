import os
from build_dataset import build_dataset
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


# define parameters
GEOMETRY_NAME           = 'circular_hole'       # This is more like a dataset name
TRAIN_VAL_TEST_SPLIT    = (0.6, 0.2, 0.2)
BATCH_SIZE              = 16
NUM_WORKERS             = 0
LEARNING_RATE           = 1e-3
EPOCHS                  = 5
EVAL_ON_STEP            = 350
LOSS_PRINT_DECIMALS     = 5
SAVE_MODEL_AFTER_EPOCHS = 3

# get datasets
train_dataloader, val_dataloader, test_dataloader = build_dataset(GEOMETRY_NAME, TRAIN_VAL_TEST_SPLIT, BATCH_SIZE, NUM_WORKERS)

# choose training device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

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
    
# prepare checkpoint folder
checkpoint_folder = os.path.join('checkpoints', GEOMETRY_NAME)
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)
else:
    assert False, 'Please clear existing checkpoints'

# create model and send to device
model = NeuralNetwork().to(device)
print('\n=== Model structure ===')
print(model, '\n')

# save model structure to checkpoints
with open(os.path.join(checkpoint_folder, 'model_structure.txt'), "w") as file:
    file.write(str(model))

# create optimizer and loss
opt = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# create tensorboard writer
writer = SummaryWriter(f'runs/{GEOMETRY_NAME}')

print(f'Nr of training batches: {len(train_dataloader)}')
print(f'Nr of validation batches: {len(val_dataloader)}')
print()

# Training flow
for epoch in range(EPOCHS):
    running_loss = 0.0
    
    for i, batch in enumerate(train_dataloader):
        # basic training loop
        inputs, labels = batch
        inputs = inputs.to(device, dtype=torch.float32) # float32 necessary on mps devices
        labels = labels.to(device, dtype=torch.float32)
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        
        running_loss += loss.item()
        if i % EVAL_ON_STEP == EVAL_ON_STEP-1: # every n mini-batches
            
            # validation
            running_vloss = 0.0
            
            model.train(False)
            for j, vbatch in enumerate(test_dataloader):
                vinputs, vlabels = vbatch
                vinputs = vinputs.to(device, dtype=torch.float32)
                vlabels = vlabels.to(device, dtype=torch.float32)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
            model.train(True)
            
            avg_loss = running_loss / EVAL_ON_STEP
            avg_vloss = running_vloss / len(test_dataloader)
            
            print(f'Epoch {epoch:4d} Batch {i+1:6d} Loss {avg_loss:{LOSS_PRINT_DECIMALS+3}.{LOSS_PRINT_DECIMALS}f} Vloss {avg_vloss:{LOSS_PRINT_DECIMALS+3}.{LOSS_PRINT_DECIMALS}f}')
            
            # log the running loss average
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch * len(train_dataloader) + i)
            
            running_loss = 0.0
    
    # save model parameters every n epochs
    if epoch % SAVE_MODEL_AFTER_EPOCHS == SAVE_MODEL_AFTER_EPOCHS-1:
        torch.save(model.state_dict(), os.path.join(checkpoint_folder, f'checkpoint_epoch_{epoch}.pt'))

# save final model
torch.save(model.state_dict(), os.path.join(checkpoint_folder, f'checkpoint_epoch_{epoch}.pt'))

print ('\nTraining finished')

# make sure everything is written to tensorboard
writer.flush()