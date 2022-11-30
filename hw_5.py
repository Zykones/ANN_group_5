import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

#load cifar 10 dataset
train_ds, test_ds = tfds.load("cifar10", split=["train", "test"], as_supervised=True)

def prepare_data(cifar10):
    #flatten 32x32 vectors
    cifar10 = cifar10.map(lambda img, target: (tf.reshape(img, (-1.)), target))
    
    #unit8 to float32 ?
    cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))

    #normalization
    cifar10 = cifar10.map(lambda img, target: ((img/128.0)-1.0, target))

    #one hot ?
    #welche depth?
    cifar10 = cifar10.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    #cache to memory
    cifar10 = cifar10.cache()

    #shuffle, batch, prefetch
    #Werte anpassen?
    cifar10 = cifar10.shuffle(10000)
    cifar10 = cifar10.batch(32)
    cifar10 = cifar10.prefetch(20)

    return cifar10


def Cifar10Model(tf.keras.Model):
    def __init__(self, neurons):
        super(Cifar10Model, self).__init__()
        self.convlayer1 = tf.keras.layers.ConV2D(filters=24, kernel_size=3, 
                                                    padding="same", activation="relu")
        self.convlayer2 = tf.keras.layers.ConV2D(filters=24, kernel_size=3,
                                                    padding="same", activation="relu")
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

        self.out = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, x):
        x = self.conlayer1(x)
        x = self.convlayer2(x)
        x = self.pooling(x)
        x = self.out(x)
        return x


def train_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss


def test(model, test_data, loss_function):
    test_accuracy_aggregator = []
    test_loss_aggregator = []

    #eventuell numpy functions mit tf function ersetzen weil die schneller sind, ist aber nicht zwingend nötig

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    
    return test_loss, test_accuracy


#hyperparameters and variables for training loop
epochs = 2
eta = 0.001 #ist der Wert immer noch okay?
num_neurons_hidden_layer = 30 #sind das immer noch hidden layers und brauchen wir 30?
optimizer = tf.keras.optimizers.Adam(learning_rate=eta)

model = Cifar10Model(num_neurons_hidden_layer) #neurons?
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
train_losses = []
test_losses = []
test_accs =[]


#training loop
for epoch in range(epochs):
    print(f"Epoch: {str(epoch)}")

    epoch_loss_agg = []

    #was ist mit train_dataset und test_dataset? wie heißen die jetzt bzw wo kriegen wir die her?

    for input, target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)

    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accs.append(test_accuracy)