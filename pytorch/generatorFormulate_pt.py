import numpy as np
import torch
from torch import nn
from torch import autograd
from torch import optim
import h5py

def norm_data(x, num_sample, num_rx, num_tx, num_delay):
    x2 = np.reshape(x, [num_sample, num_rx * num_tx * num_delay * 2])
    x_max = np.max(abs(x2), axis=1)
    x_max = x_max[:,np.newaxis]
    x3 = x2 / x_max
    y = np.reshape(x3, [num_sample, 1, num_rx * num_tx , num_delay * 2])
    return y
NUM_RX = 4
NUM_TX = 32
NUM_DELAY = 32
NUM_SAMPLE_TRAIN = 500
LATENT_DIM = 128
BATCH_SIZE = 256
EPOCH = 20

data_train = h5py.File('H1_32T4R.mat', 'r')
data_train = np.transpose(data_train['H1_32T4R'][:])  # (500,4,32,32)
data_train = data_train[:, :, :, :, np.newaxis]
data_train = np.concatenate([data_train['real'], data_train['imag']], 4)
data_train = np.reshape(data_train, [NUM_SAMPLE_TRAIN, NUM_RX* NUM_TX, NUM_DELAY* 2, 1])  # (500,128,64,1)
train_channel = norm_data(data_train, NUM_SAMPLE_TRAIN, NUM_RX, NUM_TX, NUM_DELAY)  # (500,1,128,64)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        zero_padding = nn.Sequential(
            nn.ZeroPad2d((0, 0, 32, 32)),
        )
        conv_block = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 2, padding=1),
        )
        linear_block = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.zero_padding = zero_padding
        self.conv_block = conv_block
        self.linear_block = linear_block
    def forward(self, input):
        output = self.zero_padding(input)
        output = self.conv_block(output)
        output = output.view(-1, 64)
        output = self.linear_block(output)
        return output

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        linear_block = nn.Sequential(
            nn.Linear(LATENT_DIM, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        conv_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, 2, stride=2),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4, 1, 2, stride=2),
            nn.Tanh()
        )
        self.linear_block = linear_block
        self.conv_block = conv_block
    def forward(self, input):
        output = self.linear_block(input)
        output = output.view(-1, 64, 4, 2)
        output = self.conv_block(output)
        return output

g_model = Generator()
d_model = Discriminator()

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 3
if use_cuda:
    d_model = d_model.cuda(gpu)
    g_model = g_model.cuda(gpu)

one = torch.tensor(1, dtype=torch.float)
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

discriminator_optimizer = optim.Adam(d_model.parameters(), lr=2e-4, betas=(0.5, 0.9))
generator_optimizer = optim.Adam(g_model.parameters(), lr=2e-4, betas=(0.5, 0.9))

def cal_gradient_penalty(d_model, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous().view(BATCH_SIZE, 1, 128, 64)
    alpha = alpha.cuda(gpu) if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = d_model(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

for iteration in range(int(EPOCH*(NUM_SAMPLE_TRAIN/BATCH_SIZE))):
    for p in d_model.parameters():
        p.requires_grad = True
    for i in range(3):
        d_model.zero_grad()
        real_channel = train_channel[np.random.choice(train_channel.shape[0], BATCH_SIZE, replace=False)]
        real_channel = torch.from_numpy(real_channel)
        if use_cuda:
            real_channel = real_channel.cuda(gpu)
        real_channel_v = autograd.Variable(real_channel)
        real_logits = d_model(real_channel_v)
        real_logits = real_logits.mean()
        real_logits.backward(mone)
        random_latent_vectors = torch.randn(BATCH_SIZE, LATENT_DIM)
        if use_cuda:
            random_latent_vectors = random_latent_vectors.cuda(gpu)
        with torch.no_grad():
            noisev = autograd.Variable(random_latent_vectors)
        fake_channel = autograd.Variable(g_model(noisev).data)
        fake_logits = d_model(fake_channel)
        fake_logits = fake_logits.mean()
        fake_logits.backward(one)
        gradient_penalty = cal_gradient_penalty(d_model, real_channel_v.data, fake_channel.data)
        gradient_penalty.backward()
        d_loss = fake_logits - real_logits + gradient_penalty
        discriminator_optimizer.step()

    for p in d_model.parameters():
        p.requires_grad = False
    g_model.zero_grad()
    random_latent_vectors = torch.randn(BATCH_SIZE, LATENT_DIM)
    if use_cuda:
        random_latent_vectors = random_latent_vectors.cuda(gpu)
    noisev = autograd.Variable(random_latent_vectors)
    fake_channel = g_model(noisev)
    fake_logits = d_model(fake_channel)
    fake_logits = fake_logits.mean()
    fake_logits.backward(mone)
    g_loss = -fake_logits
    generator_optimizer.step()

    if iteration % (int(NUM_SAMPLE_TRAIN/BATCH_SIZE)) == 0:
        torch.save(g_model, 'generator.pth.tar')
        print('Epoch = ' + str(int(iteration / int(NUM_SAMPLE_TRAIN/BATCH_SIZE))) + ', d_loss = ' + str(d_loss.cpu().data.numpy()) + ', g_loss = ' + str(g_loss.cpu().data.numpy()))


