#=======================================================================================================================
#=======================================================================================================================
# Package Importing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
#=======================================================================================================================
#=======================================================================================================================
# Utils
def norm_data(x, num_sample, num_rx, num_tx, num_delay):
    x2 = np.reshape(x, [num_sample, num_rx * num_tx * num_delay * 2])
    x_max = np.max(abs(x2), axis=1)
    x_max = x_max[:,np.newaxis]
    x3 = x2 / x_max
    y = np.reshape(x3, [num_sample, num_rx * num_tx , num_delay * 2, 1])
    return y
def conv_block(x, filters, activation, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True, use_bn=False, use_dropout=False, drop_value=0.5,):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x
def upsample_block(x,filters,activation,kernel_size=(3, 3),strides=(1, 1),up_size=(2, 2),padding="same",use_bn=False,use_bias=True,use_dropout=False,drop_value=0.3,):
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)
#=======================================================================================================================
#=======================================================================================================================
# Parameter Setting
NUM_RX = 4
NUM_TX = 32
NUM_DELAY = 32
NUM_SAMPLE_ALL = 500
NUM_SAMPLE_TRAIN = 500
IMG_SHAPE = (128, 64, 1)
LATENT_DIM = 128
BATCH_SIZE = 256
# Data Loading
print('Data Loading...')
data_train = h5py.File('H1_32T4R.mat', 'r')
data_train = np.transpose(data_train['H1_32T4R'][:])
data_train = data_train[:, :, :, :, np.newaxis]
data_train = np.concatenate([data_train['real'], data_train['imag']], 4)
data_train = np.reshape(data_train, [NUM_SAMPLE_TRAIN, NUM_RX* NUM_TX, NUM_DELAY* 2, 1])
train_channel = norm_data(data_train, NUM_SAMPLE_TRAIN, NUM_RX, NUM_TX, NUM_DELAY)
#=======================================================================================================================
#=======================================================================================================================
# Formulating
def Discriminator():
    img_input = layers.Input(shape=IMG_SHAPE)
    x = layers.ZeroPadding2D((0, 32))(img_input)
    x = conv_block(x,8,kernel_size=(3, 3),strides=(2, 2),use_bn=False,activation=layers.LeakyReLU(0.2),use_bias=True,use_dropout=False,drop_value=0.3,)
    x = conv_block(x,16,kernel_size=(3, 3),strides=(2, 2),use_bn=False,activation=layers.LeakyReLU(0.2),use_bias=True,use_dropout=True,drop_value=0.3,)
    x = conv_block(x,32,kernel_size=(3, 3),strides=(2, 2),use_bn=False,activation=layers.LeakyReLU(0.2),use_bias=True,use_dropout=True,drop_value=0.3,)
    x = conv_block(x,64,kernel_size=(3, 3),strides=(2, 2),use_bn=False,activation=layers.LeakyReLU(0.2),use_bias=True,use_dropout=False,drop_value=0.3,)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model
d_model = Discriminator()
d_model.summary()

def Generator():
    noise = layers.Input(shape=(LATENT_DIM,))
    x = layers.Dense(4 * 4 * 64, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((4, 4, 64))(x)
    x = upsample_block(x,64,layers.LeakyReLU(0.2),strides=(1, 1),use_bias=False,use_bn=True,padding="same",use_dropout=False,)
    x = upsample_block(x,32,layers.LeakyReLU(0.2),strides=(1, 1),use_bias=False,use_bn=True,padding="same",use_dropout=False,)
    x = upsample_block(x,16,layers.LeakyReLU(0.2),strides=(1, 1),use_bias=False,use_bn=True,padding="same",use_dropout=False,)
    x = upsample_block(x,8,layers.LeakyReLU(0.2),strides=(1, 1),use_bias=False,use_bn=True,padding="same",use_dropout=False,)
    x = upsample_block(x, 1, layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True)
    x = layers.Cropping2D((0, 32))(x)
    g_model = keras.models.Model(noise, x, name="generator")
    return g_model
g_model = Generator()
g_model.summary()

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_channel, fake_channel):
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_channel - real_channel
        interpolated = real_channel + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_channel):
        if isinstance(real_channel, tuple):
            real_channel = real_channel[0]
        batch_size = tf.shape(real_channel)[0]
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_channel = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_channel, training=True)
                real_logits = self.discriminator(real_channel, training=True)
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_channel, fake_channel)
                d_loss = d_cost + gp * self.gp_weight
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return {"d_loss": d_loss, "g_loss": g_loss}

generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

wgan = WGAN(discriminator=d_model,generator=g_model,latent_dim=LATENT_DIM,discriminator_extra_steps=3,)
wgan.compile(d_optimizer=discriminator_optimizer,g_optimizer=generator_optimizer,g_loss_fn=generator_loss,d_loss_fn=discriminator_loss,)

wgan.fit(train_channel, batch_size=BATCH_SIZE, epochs=2)
wgan.generator.save('generator.h5')