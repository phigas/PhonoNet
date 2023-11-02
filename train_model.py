from build_dataset import build_dataset
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


# define parameters
GEOMETRY_NAME   = 'circular_hole'
TRAIN_SPLIT     = 0.7
BATCH_SIZE      = 16
NUM_WORKERS     = 0
LEARNING_RATE   = 1e-3
EPOCHS          = 5
EVAL_ON_STEP    = 100


# get datasets
# torch.set_default_dtype(torch.float32) # for MPS
train_dataloader, test_dataloader = build_dataset(GEOMETRY_NAME, TRAIN_SPLIT, BATCH_SIZE, NUM_WORKERS)

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

# create model and send to device
model = NeuralNetwork().to(device)
print('Model structure')
print(model)

# create optimizer and loss
opt = Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# create tensorboard writer
writer = SummaryWriter(f'runs/{GEOMETRY_NAME}')

# Training flow
for epoch in range(EPOCHS):
    running_loss = 0.0
    
    for i, batch in enumerate(train_dataloader):
        # basic training loop
        inputs, labels = batch
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        opt.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        
        running_loss += loss.item()
        if i % EVAL_ON_STEP == EVAL_ON_STEP-1: # every 1000 mini-batches
            print('Batch {}'.format(i + 1))
            
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
            
            # log the running loss average
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch * len(train_dataloader) + i)
            
            running_loss = 0.0
            
            
print ('Training finished')

writer.flush()