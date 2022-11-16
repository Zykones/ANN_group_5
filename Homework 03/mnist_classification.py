import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class MnistModel(tf.keras.Model):
    def __init__(self, neurons):
        super(MnistModel,self).__init__()
        self.dense1= Dense(256, activation="relu") # hidden Layer 01
        self.dense2= Dense(neurons, activation="relu") #  Hidden Layer 02
        self.out = Dense(10, activation="relu") # Output layer
    
    @tf.function
    def call(self, inputs):
        x  = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)


def prepare_data(mnist):
    #flatten 28x28 to vectors
    mnist = mnist.map(lambda img, target:(tf.reshape(img, (-1,)), target))

    #unit8 to float32
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))

    #imput normaliziation: 
    mnist = mnist.map(lambda img, target: ((img/128.0)-1.0, target ))

    #one hot
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    #cache to memory 
    mnist = mnist.cache()

    #shuffle, batch, prefetch

    mnist = mnist.shuffle(10000) 
    mnist = mnist.batch(32)
    mnist = mnist.prefetch(20)

    return mnist


def train_step(model, input, target, loss_function, optimizer):
    #loss_function and optimizer are TF classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def test(model, test_data, loss_function):
    
    test_accuracy_aggregator = []
    test_loss_aggregator = []

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

def visualization (train_losses, test_losses ,test_accuracies):
    """ Visualizes accuracy and loss for training and test data using
        the mean of each epoch .
        3 Loss is displayed in a regular line , accuracy in a dotted
        line .
        4 Training data is displayed in blue , test data in red .
        5 Parameters
        6 ----------
        7 train_losses : numpy . ndarray
        8 training losses
        9 train_accuracies : numpy . ndarray
        10 training accuracies
        11 test_losses : numpy . ndarray
        12 test losses
        13 test_accuracies : numpy . ndarray
        14 test accuracies
        15 """
    plt.figure ()
    line1, = plt. plot ( train_losses , "b-")
    line2, = plt. plot ( test_losses , "r-")
    #line3, = plt. plot ( train_accuracies , "b:")
    line4, = plt. plot ( test_accuracies , "r:")
    plt.xlabel (" Training steps ")
    plt.ylabel (" Loss / Accuracy ")
    #     plt.legend (( line1 , line2 , line3 , line4 ) , ("training loss", "test loss", "train accuracy", "test accuracy"))
    plt.legend (( line1 , line2 , line4 ) , ("training loss", "test loss", "test accuracy"))
    plt.show ()

(train_ds , test_ds) , ds_info = tfds.load ("mnist", split =["train", "test"], as_supervised = True , with_info = True)

#print(ds_info)

train_dataset = train_ds.apply(prepare_data)
test_dataset = test_ds.apply(prepare_data)

tf.keras.backend.clear_session()

# Optional 
train_dataset = train_dataset.take(1000)
test_dataset = test_dataset.take(1000)
####################################################
### Hyperparameters 
epochs = 2
# Learning Rate
eta = 0.001
num_neurons_hidden_layer = 30
optimizer = tf.keras.optimizers.Adam(learning_rate=eta)
#########################################################
model = MnistModel(num_neurons_hidden_layer)
#Loss function
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()



train_losses = []

test_losses = []
test_accs = []

for epoch in range(epochs):
    print(f"Epoch: {str(epoch)}")

    epoch_loss_agg = []

    for input, target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    test_loss, test_accurary = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accs.append(test_accurary)

visualization(train_losses, test_losses, test_accs)


