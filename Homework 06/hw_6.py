import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import layers, optimizers, losses, metrics
import numpy as np
import datetime
import tqdm
import matplotlib.pyplot as plt

# Number of paralell threads, 0 0 means the system picks an appropriate number.
#############################################################################################
# if you dont know what threads are or how many your system can handle, change
# #####################
# num_threads to zero!!
#############################################################################################
num_threads = 0
##############################################################################
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

def prepare_data(cifar10, batch_size=32):
    #uint_8 to float32 conversion
    cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))
    
    #normalization
    cifar10 = cifar10.map(lambda img, target: ((img/128.0)-1.0, target))

    #one hot encoding
    cifar10 = cifar10.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    #cache to memory
    cifar10 = cifar10.cache()

    #shuffle, batch, prefetch
    #Werte anpassen?
    cifar10 = cifar10.shuffle(50000)
    cifar10 = cifar10.batch(32)
    cifar10 = cifar10.prefetch(tf.data.AUTOTUNE)

    return cifar10

def create_summary_writers(config_name="RUN"):
    
    # Define where to save the logs
    # along with this, you may want to save a config file with the same name so you know what the hyperparameters were used
    # alternatively make a copy of the code that is used for later reference
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_path = f"Homework 05/logs/{config_name}/{current_time}/train"
    val_log_path = f"Homework 05/logs/{config_name}/{current_time}/val"

    # log writer for training metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_path)

    # log writer for validation metrics
    val_summary_writer = tf.summary.create_file_writer(val_log_path)
    
    return train_summary_writer, val_summary_writer

def training_loop(model, train_ds, val_ds, epochs:int, train_summery_writer, val_sumnmery_writer,  save_path=None) -> dict:
    """Implements the training loop"""
    value_dict = {"loss": [], "acc": [], "val_loss": [], "val_acc":[]}
    for e in range(epochs):

        for data in tqdm.tqdm(train_ds, position=0, leave=True):
            metrics = model.train_step(data)
        
        with train_summary_writer.as_default():
        # for scalar metrics:
            for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=e)
            # alternatively, log metrics individually (allows for non-scalar metrics such as tf.keras.metrics.MeanTensor)
            # e.g. tf.summary.image(name="mean_activation_layer3", data = metrics["mean_activation_layer3"],step=e)
        
        #print the metrics
        for (key, value) in metrics.items():
            value_dict[key].append(value.numpy())
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])

        model.reset_metrics()

        for data in val_ds:
            metrics = model.test_step(data)

        with val_summary_writer.as_default():
            # for scalar metrics:
            for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=e)
            # alternatively, log metrics individually (allows for non-scalar metrics such as tf.keras.metrics.MeanTensor)
            # e.g. tf.summary.image(name="mean_activation_layer3", data = metrics["mean_activation_layer3"],step=e)
            
        for (key, value) in metrics.items():
            value_dict["val_"+key].append(value.numpy())

        print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])
        # 7. reset metric objects
        model.reset_metrics()
    
    if save_path:
        model.save_weights(save_path)
    return value_dict

def visualise(values:dict) -> None:
    """Creates two Subplots with Losses and Accurarcys in them and shows the plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("Loss and Accurarcy Curves")

    ax1.plot(range(epochs), values["loss"], color="blue")
    ax1.plot(range(epochs), values["val_loss"], color="green")
    ax1.set_title("Losses")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.grid(True)
    ax1.legend(["Training", "Validation"])


    ax2.plot(range(epochs), values["acc"], color="blue")
    ax2.plot(range(epochs), values["val_acc"], color="green")
    ax2.set_title("Accurarcys")
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.grid(True)
    ax2.legend(["Training", "Validation"])

    plt.show()

    




class Cifar10Model(tf.keras.Model):
    def __init__(self, neurons:int, eta:float=0.001 ) -> None:
        super().__init__()
        
        # Layers
        self.convlayer1 = layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3))
        self.batch_norm1 = layers.BatchNormalization() #new 
        self.convlayer2 = layers.Conv2D(32, (3,3), padding='same', activation='relu')
        self.batch_norm2 = layers.BatchNormalization() #new
        self.pooling1 = layers.MaxPooling2D(pool_size=(2,2))
        self.convlayer3 = layers.Conv2D(64, (3,3), padding='same', activation='relu')
        self.convlayer4 = layers.Conv2D(64, (3,3), padding='same', activation='relu')
        self.pooling2 = layers.MaxPooling2D(pool_size=(2,2))
        self.convlayer5 = layers.Conv2D(128, (3,3), padding="same", activation="relu") #new
        self.global_pool = layers.GlobalAvgPool2D()

        self.dense = layers.Dense(neurons, activation="relu")

        self.out = layers.Dense(10, activation="softmax")

        # Metrics Object 
        self.metrics_list = [metrics.CategoricalCrossentropy(name="loss"), metrics.CategoricalAccuracy(name="acc")]
        
        # Optimizer
        self.optimizer = optimizers.Adam(learning_rate=eta)

        #Loss
        self.loss_function = losses.CategoricalCrossentropy()

    @tf.function
    def call(self, x):
        x = self.convlayer1(x)
        x = self.batch_norm1(x) #new 
        x = self.convlayer2(x)
        x = self.batch_norm2(x) #new
        x = self.pooling1(x)
        x = self.convlayer3(x)
        x = self.convlayer4(x)
        x = self.pooling2(x)
        x = self.convlayer5(x) #new
        x = self.global_pool(x)
        x = self.dense(x)
        x = self.out(x)
        return x
    
    # metrics property
    @property
    def metrics(self) -> list:
        """return a list with all metrics in the model"""

        return self.metrics_list
    
    def reset_metrics(self) -> None:
        """Resetting the metrics object"""

        for metrics in self.metrics:
            metrics.reset_states()


    @tf.function
    def train_step(self, train_data) -> dict:

        data, target = train_data

        with tf.GradientTape() as tape:
            prediction = self(data, training=True)
            loss = self.loss_function(target, prediction)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    	
        #update metrics according to loss
        self.metrics[0].update_state(target, prediction)
        self.metrics[1].update_state(target, prediction)
        
        return {m.name : m.result() for m in self.metrics}

    @tf.function
    def test_step(self, test_data):
        data, target = test_data

        prediction = self(data)
        loss = self.loss_function(target, prediction)

        self.metrics[0].update_state(target, prediction)
        self.metrics[1].update_state(target, prediction)

        return {m.name : m.result() for m in self.metrics}



#load cifar 10 dataset
train_ds, test_ds = tfds.load("cifar10", split=["train", "test"], as_supervised=True)

train_ds = prepare_data(train_ds)
test_ds = prepare_data(test_ds)

#hyperparameters and variables for training loop
################################################
epochs = 10 #from 2 to 10 
eta = 0.001 
num_neurons_hidden_layer = 30
################################################


train_summary_writer, val_summary_writer = create_summary_writers(config_name="Adam")

model = Cifar10Model(num_neurons_hidden_layer, eta)

val_dict = training_loop(model, train_ds, test_ds, epochs, train_summary_writer, val_summary_writer)

visualise(val_dict)


