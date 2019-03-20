import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import random
import matplotlib.pyplot as plt
from functions import *

# region Pytorch Init
print("Using Pytorch " + torch.__version__)
print(f"{torch.cuda.device_count()} CUDA device(s) found")
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())

    print("Using CUDA " + torch.version.cuda + " - " + device_name)
else:
    print("Using CPU")
    device = torch.device('cpu')

# endregion
# region Import dataset

# helper function to make getting another batch of data easier
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
class_horses = 7
class_birds = 2

# Get combined dataset
print("Loading dataset")
dataset_train = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
]))
dataset_test = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
]))
dataset = torch.utils.data.ConcatDataset((dataset_train, dataset_test))

# Get dataset of horses/birds
print("Converting dataset to just be horses/birds")
dataset_birds = list(x for x in dataset if x[1] == class_birds)
dataset_horses = list(x for x in dataset if x[1] == class_horses)

dataset_hb = []
for i in dataset:
    if i[1] == class_horses:
        dataset_hb.append(i)

train_loader = torch.utils.data.DataLoader(dataset_hb, shuffle=True, batch_size=16, drop_last=True)

train_iterator = iter(cycle(train_loader))

print(f'> Size of dataset (training + test): {len(train_loader.dataset):,}')

# endregion
# region Define a simple model

# define the model (a simple autoencoder)
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4, stride=1, padding=0, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.lin1 = nn.Linear(in_features=64 * 9 * 9, out_features=400)
        self.lin2 = nn.Linear(in_features=400, out_features=30)

        # Decoder

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x

    # encode (flatten as linear, then run first half of network)
    def encode(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # print(x.shape)

        x = x.view(x.size(0), -1)  # flatten input as we're using linear layers
        # print(x.shape)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x

    # decode (run second half of network then unflatten)
    def decode(self, x):
        for i in range(4, 8):
            x = self.layers[i](x)
        x = x.view(x.size(0), 3, 32, 32)
        return x

N = MyNetwork().to(device)

print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(N.parameters())):,}')

# initialise the optimiser
optimiser = torch.optim.Adam(N.parameters(), lr=0.001)
epoch = 0

# endregion
# region Main training loop
start_time = time.time()
loop_start_time = time.time()

train_acc_graph = []
test_acc_graph = []
train_loss_graph = []
test_loss_graph = []

best_loss = 1000
best_epoch = 0

# training loop, feel free to also train on the test dataset if you like for generating the pegasus
while epoch < 10:
    # arrays for metrics
    logs = {}
    train_loss_arr = np.zeros(0)

    # iterate over some of the train dataset
    for x, _ in train_loader:
        x = x.to(device)

        optimiser.zero_grad()
        p = N(x)
        loss = ((p - x) ** 2).mean()  # simple l2 loss
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)

    total_duration = time.time() - start_time
    loop_duration = time.time() - loop_start_time

    print(f"Epoch {epoch + 1} finished ({format_time(loop_duration)}/{format_time(total_duration)})")
    print("\tLoss: " + format_acc(train_loss_arr.mean()))
    train_loss_graph.append(train_loss_arr.mean())

    if train_loss_arr.mean() < best_loss:
        best_loss = train_loss_arr.mean()
        best_epoch = epoch

    epoch += 1
    loop_start_time = time.time()

print(f"Best train loss occurred in epoch {best_epoch + 1}: " + format_acc(best_loss, convert_to_percentage=True))

# endregion
# region Generate a pegasus
"""**Generate a Pegasus by interpolating between the latent space encodings of a horse and a bird**"""

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])

    horse = random.choice(dataset_horses)[0].to(device)  # horse
    # example_2 = test_loader.dataset[160][0].to(device)  # bird

    horse_encoded = N.encode(horse.unsqueeze(0))
    # example_2_code = N.encode(example_2.unsqueeze(0))

    # this is some sad blurry excuse of a Pegasus, hopefully you can make a better one
    # bad_pegasus = N.decode(0.9 * example_1_code + 0.1 * example_2_code).squeeze(0)
    bad_pegasus = N.decode(0.9 * horse_encoded).squeeze(0)

    plt.grid(False)
    plt.imshow(bad_pegasus.cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0), cmap=plt.cm.binary)

plt.show()

# endregion

# for i in range(len(test_loader.dataset.test_labels)):
#  print(class_names[test_loader.dataset.test_labels[i]] + '\t idx: ' + str(i))
