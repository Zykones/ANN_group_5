import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 128
num_epochs = 10


# Download the data
data = np.load("C:\Users\henni\Documents\GitHub\ANN_group_5\Homework 09\data\candles.npy")

# Reshape the images to (28, 28, 1)
data = data.reshape(-1, 28, 28, 1)

# Normalize the images
data = data / 255

# Create a tf.data.Dataset object
dataset = tf.data.Dataset.from_tensor_slices(data)

# Perform other necessary processing steps (batching, shuffling, etc)
dataset = dataset.batch(batch_size).shuffle(buffer_size=1000)

#The model
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, input_shape=(28, 28, 1))
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, training=True):
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        x = tf.nn.leaky_relu(x)
        x = self.dropout(x, training=training)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = tf.keras.layers.Dense(7 * 7 * 64, input_shape=(100,))
        self.reshape = tf.keras.layers.Reshape((7, 7, 64))
        self.conv1 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2)
        self.conv2 = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, activation='tanh')

    def call(self, x, training=True):
        x = self.dense(x)
        x = tf.nn.leaky_relu(x)
        x = self.reshape(x)
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        return x


#Training
# Define the loss functions and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy()
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# Define the training loop
for epoch in range(num_epochs):
    for step, real_images in enumerate(dataset):
        # Generate fake images
        random_noise = tf.random.normal([batch_size, 100])
        fake_images = Generator(random_noise, training=True)

        # Compute the loss of the discriminator on real and fake images
        real_logits = Discriminator(real_images, training=True)
        fake_logits = Discriminator(fake_images, training=True)
        d_loss_real = cross_entropy(tf.ones_like(real_logits), real_logits)
        d_loss_fake = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
        d_loss = d_loss_real + d_loss_fake

        # Backpropagate the discriminator's loss and update its parameters
        d_optimizer.minimize(d_loss, Discriminator.trainable_variables)

        # Generate new random noise and compute the loss of the generator
        random_noise = tf.random.normal([batch_size, 100])
        fake_images = Generator(random_noise, training=True)
        fake_logits = Discriminator(fake_images, training=True)
        g_loss = cross_entropy(tf.ones_like(fake_logits), fake_logits)

        # Backpropagate the generator's loss and update its parameters
        g_optimizer.minimize(g_loss, Generator.trainable_variables)

    # Print the current losses
    print("Epoch: {}, Discriminator Loss: {}, Generator Loss: {}".format(epoch, d_loss, g_loss))

# Visualize the generated images
random_noise = tf.random.normal([batch_size, 100])
generated_images = Generator(random_noise, training=False)

# display candle 
fig = plt.figure(figsize=(6, 6))
for j in range(16):
    plt.subplot(4, 4, j+1)
    noise = tf.random.normal([1, 100])
    generated_images = Generator(noise, training=False)
    plt.imshow(generated_images[0, :, :, 0] * 127.5 + 127.5, cmap="gray_r")
    plt.axis('off')
plt.show()