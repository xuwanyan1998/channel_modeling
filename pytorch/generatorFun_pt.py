# Package Importing
import numpy as np
import torch
from torch import nn

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

def generator_1(num_fake_1, file_generator_1, file_real_1):
    use_cuda = torch.cuda.is_available()
    generator_C = torch.load(file_generator_1)
    generator_C = generator_C.cuda().eval()
    # real_C = np.load(file_real_1)
    num_tx = 32
    num_rx = 4
    num_delay = 32
    latent_dim = 128
    size_packet = 500
    with torch.no_grad():
        for idx in range(int(num_fake_1 / size_packet)):
            latent_vectors = torch.randn(size_packet, latent_dim)
            if use_cuda:
                latent_vectors = latent_vectors.cuda()
            fake_data = generator_C(latent_vectors)
            fake_data = fake_data.cpu().numpy()
            fake_data = np.reshape(fake_data, [size_packet, num_rx, num_tx, num_delay, 2])
            fake_data_r = fake_data[:, :, :, :, 0]
            fake_data_i = fake_data[:, :, :, :, 1]
            fake_data_reshape = fake_data_r + fake_data_i * 1j
            if idx == 0:
                data_fake_all = fake_data_reshape
            else:
                data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
    return data_fake_all

def generator_2(num_fake_2, file_generator_2, file_real_2):
    use_cuda = torch.cuda.is_available()
    generator_U = torch.load(file_generator_2)
    generator_U = generator_U.cuda()
    generator_U.eval()
    # real_U = np.load(file_real_2)
    num_tx = 32
    num_rx = 4
    num_delay = 32
    latent_dim = 128
    size_packet = 500
    with torch.no_grad():
        for idx in range(int(num_fake_2 / size_packet)):
            latent_vectors = torch.randn(size_packet, latent_dim)
            if use_cuda:
                latent_vectors = latent_vectors.cuda()
            fake_data = generator_U(latent_vectors)
            fake_data = fake_data.cpu().numpy()
            fake_data = np.reshape(fake_data, [size_packet, num_rx, num_tx, num_delay, 2])
            fake_data_r = fake_data[:, :, :, :, 0]
            fake_data_i = fake_data[:, :, :, :, 1]
            fake_data_reshape = fake_data_r + fake_data_i * 1j
            if idx == 0:
                data_fake_all = fake_data_reshape
            else:
                data_fake_all = np.concatenate((data_fake_all, fake_data_reshape), axis=0)
    return data_fake_all
#=======================================================================================================================
#=======================================================================================================================