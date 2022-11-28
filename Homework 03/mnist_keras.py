import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

######################################################
n_epochs = 10
eta = 0.001
n_neurons_hidden_layer = 30
n_neurons_hidden_layer2 = 256
batch_size = 32

######################################################

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() #60000 images for train, 10000 images for test
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), #input layer with 28x28 = 784 neurons
  tf.keras.layers.Dense(n_neurons_hidden_layer2, activation='relu'),
  tf.keras.layers.Dense(n_neurons_hidden_layer, activation='relu'), #Dense hidden layer with n_neuron_hidden_layer neurons
  tf.keras.layers.Dense(10) #Output layer without Softmax!
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = keras.optimizers.Adam(learning_rate=eta)

model.compile(optimizer=opt,
              loss=loss_fn,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(x_test, y_test)) #Train and Test 
model.evaluate(x_test,  y_test, verbose=2)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('model performance')
plt.ylabel('Value')
plt.xlabel('Epoch')
plt.legend(['Train loss', 'Train accuracy', 'Test loss', 'Test accuracy'], loc='upper left')
plt.show()