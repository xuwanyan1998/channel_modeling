
#=======================================================================================================================
#=======================================================================================================================
# Package Importing
import numpy as np
import tensorflow as tf
from tensorflow import keras

#=======================================================================================================================
#=======================================================================================================================
def generator_1(num_fake_1, file_generator_1, file_real_1):
    generator_C = tf.keras.models.load_model(file_generator_1)
    # real_C = np.load(file_real_1)
    num_tx = 32
    num_rx = 4
    num_delay = 32
    latent_dim = 128
    size_packet = 500
    for idx in range(int(num_fake_1 / size_packet)):
        latent_vectors = tf.random.normal(shape=(size_packet, latent_dim))
        fake_data = generator_C(latent_vectors)
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
    generator_U = tf.keras.models.load_model(file_generator_2)
    # real_U = np.load(file_real_2)
    num_tx = 32
    num_rx = 4
    num_delay = 32
    latent_dim = 128
    size_packet = 1000
    for idx in range(int(num_fake_2 / size_packet)):
        latent_vectors = tf.random.normal(shape=(size_packet, latent_dim))
        fake_data = generator_U(latent_vectors)
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