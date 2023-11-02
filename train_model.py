from build_dataset import build_dataset
import torch
from torch import nn
from torch.optim import Adam


# define parameters
GEOMETRY_NAME   = 'circular_hole'
TRAIN_SPLIT     = 0.7
BATCH_SIZE      = 4
NUM_WORKERS     = 0
LEARNING_RATE   = 1e-3


# get datasets
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
            nn.Linear(9, 512),
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

# Training flow
