import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time
import random
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul
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
class_birds = 0  # Actually an airplane!

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

dataset_birds = []
dataset_horses = []
dataset_hb = []

for i in dataset:
    if i[1] == class_birds:
        dataset_hb.append(i)
        dataset_birds.append(i)
    if i[1] == class_horses:
        dataset_hb.append(i)
        dataset_horses.append(i)

print(f"Number of birds: {len(dataset_birds):,}")
print(f"Number of horses: {len(dataset_horses):,}")

train_loader = torch.utils.data.DataLoader(dataset_hb, shuffle=True, batch_size=16, drop_last=True)

train_iterator = iter(cycle(train_loader))

print(f'> Size of dataset (training + test): {len(train_loader.dataset):,}')

# endregion
# region Define a simple model

# define the model (a simple autoencoder)
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()

        # Tensor size of the convolution output
        self.conv_size = [64, 10, 10]
        self.conv_size_prod = reduce(mul, self.conv_size)

        # Linear layer in/out size
        initial_features = 400
        reduced_features = 25

        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1)

        self.lin1 = nn.Linear(in_features=self.conv_size_prod, out_features=initial_features)
        self.lin2a = nn.Linear(in_features=initial_features, out_features=reduced_features)
        self.lin2b = nn.Linear(in_features=initial_features, out_features=reduced_features)

        # Decoder
        self.lin3 = nn.Linear(in_features=reduced_features, out_features=initial_features)
        self.lin4 = nn.Linear(in_features=initial_features, out_features=self.conv_size_prod)

        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=5, stride=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    # encode (flatten as linear, then run first half of network)
    def encode(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # print(x.shape)

        x = x.view(x.size(0), -1)  # flatten input as we're using linear layers
        # print(x.shape)
        x = F.relu(self.lin1(x))
        ret1 = self.lin2a(x)
        ret2 = self.lin2b(x)

        return ret1, ret2

    # decode (run second half of network then unflatten)
    def decode(self, x):
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        # print(x.shape)

        x = x.view(x.size(0), self.conv_size[0], self.conv_size[1], self.conv_size[2])

        # print(x.shape)
        x = F.relu(self.deconv4(x))
        # print(x.shape)
        x = F.relu(self.deconv3(x))
        # print(x.shape)
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv1(x))
        # print("Decoded")

        return x

# VAE loss has a reconstruction term and a KL divergence term summed over all elements and the batch
def vae_loss(p, x, mu, logvar, kl_weight):
    BCE = F.binary_cross_entropy(p.view(-1, 32 * 32 * 3), x.view(-1, 32 * 32 * 3), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + kl_weight * KLD

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

best_loss = 1_000_000
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

        p, mu, logvar = N(x)
        loss = vae_loss(p, x, mu, logvar, epoch / 10)
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

print(f"Best train loss occurred in epoch {best_epoch + 1}: " + format_acc(best_loss))

# endregion
# region Generate a pegasus
"""**Generate a Pegasus by interpolating between the latent space encodings of a horse and a bird**"""

plt.figure(figsize=(10, 20))
for i in range(50):
    plt.subplot(10, 5, i+1)
    plt.xticks([])
    plt.yticks([])

    horse = random.choice(dataset_horses)[0].to(device)  # horse
    bird = random.choice(dataset_birds)[0].to(device)  # bird

    horse_encoded = N.encode(horse.unsqueeze(0))[0]
    bird_encoded = N.encode(bird.unsqueeze(0))[0]

    # Create pegasus
    pegasus = N.decode(horse_encoded * 0.8 + bird_encoded * 0.4).squeeze(0)
    # pegasus = N.decode(horse_encoded * (i + 1) / 25 + bird_encoded * (24 - i) / 25).squeeze(0)
    # pegasus = N.decode(horse_encoded).squeeze(0)

    plt.grid(False)
    plt.imshow(pegasus.cpu().data.permute(0, 2, 1).contiguous().permute(2, 1, 0), cmap=plt.cm.binary)

plt.show()

# endregion

# for i in range(len(test_loader.dataset.test_labels)):
#  print(class_names[test_loader.dataset.test_labels[i]] + '\t idx: ' + str(i))
