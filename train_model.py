import os
from preprocess import load_datasets
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


def save_model(model, epoch, optimizer, criterion, checkpoint_folder):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion}, os.path.join(checkpoint_folder, f'checkpoint_epoch_{epoch}.pt'))
    
def load_model(checkpoint_folder):
    # find the latest checkpoint
    files = os.listdir(checkpoint_folder)
    checkpoint_files = [file for file in files if file.startswith('checkpoint_epoch_') and file.endswith('.pt')]
    assert checkpoint_files, 'No checkpoints found'
    epoch_numbers = [int(file.split('_')[-1].split('.')[0]) for file in checkpoint_files]
    highest_epoch = max(epoch_numbers)
    
    # load checkpoint
    checkpoint = torch.load(os.path.join(checkpoint_folder, f'checkpoint_epoch_{highest_epoch}.pt'))
    return checkpoint


# define parameters
GEOMETRY_NAME           = 'circular_hole'
MODEL_NAME              = 'first_test'
LEARNING_RATE           = 8e-4
EPOCHS                  = 3
EVAL_ON_STEP            = 350
LOSS_PRINT_DECIMALS     = 5
SAVE_MODEL_AFTER_EPOCHS = 3

# get datasets
train_dataloader, val_dataloader, test_dataloader = load_datasets(GEOMETRY_NAME)

# choose training device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" # mac devices
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
checkpoint_folder = os.path.join('checkpoints', GEOMETRY_NAME, MODEL_NAME)
continue_training = False
if not os.path.exists(checkpoint_folder):
    os.makedirs(checkpoint_folder)
else:
    print('=== Checkpoint folder already exists. Unless you want to continue training delete it or choose another model name.')
    response = input('Do you want to continue training y/n?\n')
    if response not in ['y', 'Y']:
        exit
    else:
        continue_training = True

# create model and send to device
model = NeuralNetwork().to(device)
print('\n=== Model structure ===')
print(model, '\n')

# handle model structure
if continue_training:
    # compare saved model structure to loaded
    with open(os.path.join(checkpoint_folder, 'model_structure.txt'), "r") as file:
        assert file.read() == str(model), 'Make sure the model has the same structure'
else:
    # save model structure
    with open(os.path.join(checkpoint_folder, 'model_structure.txt'), "w") as file:
        file.write(str(model))

# load model weights
if continue_training:
    checkpoint = load_model(checkpoint_folder)
    model.load_state_dict(checkpoint['model_state_dict'])

# create optimizer and loss
opt = Adam(model.parameters(), lr=LEARNING_RATE)
if continue_training:
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_fn = checkpoint['loss']
else:
    loss_fn = nn.CrossEntropyLoss()

# create tensorboard writer
writer = SummaryWriter(os.path.join(checkpoint_folder, 'logs'))

print(f'Nr of training batches: {len(train_dataloader)}')
print(f'Nr of validation batches: {len(val_dataloader)}')
print()

# define epoch range
if continue_training:
    starting_epoch = checkpoint['epoch'] + 1
else:
    starting_epoch = 0
ending_epoch = starting_epoch + EPOCHS

# Training flow
for epoch in range(starting_epoch, ending_epoch):
    print(f'Starting epoch {epoch}')
    running_loss = 0.0
    
    # write learning rate to tensorboard
    writer.add_scalar('Learning rate', LEARNING_RATE, epoch*len(train_dataloader))
    
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
        save_model(model, epoch, opt, loss_fn, checkpoint_folder)

# save final model
save_model(model, epoch, opt, loss_fn, checkpoint_folder)

print ('\nTraining finished')

# make sure everything is written to tensorboard
writer.flush()