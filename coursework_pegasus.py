import numpy as np
import torch
import torch.nn as nn
import torchvision
import time
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

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('data', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    shuffle=True, batch_size=16, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])),
    shuffle=False, batch_size=16, drop_last=True)

train_iterator = iter(cycle(train_loader))
test_iterator = iter(cycle(test_loader))

print(f'> Size of training dataset {len(train_loader.dataset)}')
print(f'> Size of test dataset {len(test_loader.dataset)}')

# endregion
# region Define a simple model

# define the model (a simple autoencoder)
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(in_features=3*32*32, out_features=512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=512, out_features=32))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=32, out_features=512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=512, out_features=3*32*32))
        layers.append(nn.Sigmoid())
        self.layers = layers

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return x

    # encode (flatten as linear, then run first half of network)
    def encode(self, x):
        x = x.view(x.size(0), -1)
        for i in range(4):
            x = self.layers[i](x)
        return x

    # decode (run second half of network then unflatten)
    def decode(self, x):
        for i in range(4,8):
            x = self.layers[i](x)
        x = x.view(x.size(0), 3, 32, 32)
        return x

N = MyNetwork().to(device)

print(f'> Number of network parameters {len(torch.nn.utils.parameters_to_vector(N.parameters()))}')

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
    train_acc_arr = np.zeros(0)
    test_loss_arr = np.zeros(0)
    test_acc_arr = np.zeros(0)

    # iterate over some of the train dateset
    for x, _ in train_loader:
        x = x.to(device)

        optimiser.zero_grad()
        p = N(x)
        loss = ((p-x)**2).mean()  # simple l2 loss
        loss.backward()
        optimiser.step()

        train_loss_arr = np.append(train_loss_arr, loss.cpu().data)

    total_duration = time.time() - start_time
    loop_duration = time.time() - loop_start_time

    print(f"Epoch {epoch + 1} finished ({format_time(loop_duration)}/{format_time(total_duration)})")
    # print("\tAccuracy: " + format_acc(train_acc_arr.mean(), convert_to_percentage=True))
    # print("\tVal Accuracy: " + format_acc(test_acc_arr.mean(), convert_to_percentage=True))
    print("\tLoss: " + format_acc(train_loss_arr.mean()))
    # print("\tVal loss: " + format_acc(test_loss_arr.mean()))

    # train_acc_graph.append(train_acc_arr.mean() * 100)
    # test_acc_graph.append(test_acc_arr.mean() * 100)
    train_loss_graph.append(train_loss_arr.mean())
    # test_loss_graph.append(test_loss_arr.mean())

    if train_loss_arr.mean() < best_loss:
        best_loss = train_loss_arr.mean()
        best_epoch = epoch

    epoch += 1
    loop_start_time = time.time()

print(f"Best train loss occurred in epoch {best_epoch + 1}: " + format_acc(best_loss, convert_to_percentage=True))

# endregion
# region Generate a pegasus
"""**Generate a Pegasus by interpolating between the latent space encodings of a horse and a bird**"""

example_1 = test_loader.dataset[13][0].to(device)  # horse
example_2 = test_loader.dataset[160][0].to(device)  # bird

example_1_code = N.encode(example_1.unsqueeze(0))
example_2_code = N.encode(example_2.unsqueeze(0))

# this is some sad blurry excuse of a Pegasus, hopefully you can make a better one
bad_pegasus = N.decode(0.9 * example_1_code + 0.1 * example_2_code).squeeze(0)

plt.grid(False)
plt.imshow(bad_pegasus.cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0), cmap=plt.cm.binary)

# region

for i in range(len(test_loader.dataset.test_labels)):
 print(class_names[test_loader.dataset.test_labels[i]] + '\t idx: ' + str(i))

