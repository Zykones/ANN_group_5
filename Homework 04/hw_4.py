import tensorflow_datasets as tfds
import tensorflow as tf
import tqdm
import os           #do we really need this?
import datetime

mnist = tfds.load("mnist", split =["train","test"], as_supervised=True)
train_ds = mnist[0]
val_ds = mnist[1]

def prepare_data(data, batch_size=32):
    # image should be float
    data = data.map(lambda x, t: (tf.cast(x, float), t))
    # image should be flattened
    data = data.map(lambda x, t: (tf.reshape(x, (-1,)), t))
    # image vector will here have values between -1 and 1
    data = data.map(lambda x,t: ((x/128.)-1., t))
    # we want to have two mnist images in each example
    # this leads to a single example being ((x1,y1),(x2,y2))
    zipped_ds = tf.data.Dataset.zip((data.shuffle(2000), 
                                     data.shuffle(2000)))
    # map ((x1,y1),(x2,y2)) to (x1,x2, y1==y2*) *boolean
    zipped_ds = zipped_ds.map(lambda x1, x2: (x1[0], x2[0], x1[1]==x2[1]))
    # transform boolean target to int
    zipped_ds = zipped_ds.map(lambda x1, x2, t: (x1,x2, tf.cast(t, tf.int32)))
    # batch the dataset
    zipped_ds = zipped_ds.batch(batch_size)
    # prefetch
    zipped_ds = zipped_ds.prefetch(tf.data.AUTOTUNE)
    return zipped_ds

train_ds = prepare_data(train_ds, batch_size=32)
val_ds = prepare_data(val_ds, batch_size=32)

class TwinMNISTModel(tf.keras.Model):

    # 1. constructor
    def __init__(self):
        super().__init__()
        # inherit functionality from parent class

        # optimizer, loss function and metrics
        self.metrics_list = [tf.keras.metrics.BinaryAccuracy(),
                             tf.keras.metrics.Mean(name="loss")]
        
        #self.optimizer = tf.keras.optimizers.Adam()
        self.optimizer = tf.keras.optimizers.SGD()
        
        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        
        # layers to be used
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        
        self.dense3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        
        self.out_layer = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
        
           
    # 2. call method (forward computation)
    @tf.function
    def call(self, images, training=False):
        img1, img2 = images
        
        img1_x = self.dense1(img1)
        img1_x = self.dense2(img1_x)
        
        img2_x = self.dense1(img2)
        img2_x = self.dense2(img2_x)
        
        combined_x = tf.concat([img1_x, img2_x ], axis=1)
        combined_x = self.dense3(combined_x)
        
        return self.out_layer(combined_x)
    

    # 3. metrics property
    @property
    def metrics(self):
        return self.metrics_list
        # return a list with all metrics in the model


    # 4. reset all metrics objects
    def reset_metrics(self):
        for metrics in self.metrics:
            metrics.reset_states()


    # 5. train step method
    def train_step(self, data):
        img1, img2, label = data
        
        with tf.GradientTape() as tape:
            output = self((img1, img2), training=True)
            loss = self.loss_function(label, output)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        #update metrics according to loss
        self.metrics[0].update_state(label, output)
        self.metrics[1].update_state(loss)

        # What exactly does the "mean" loss capture in this model? 
        
        # return a dictionary with metric names as keys and metric results as values
        return {m.name : m.result() for m in self.metrics}
       

    # 6. test_step method
    @tf.function
    def test_step(self, data):
        # same as train step (without parameter updates)
        img1, img2, label = data
        
        output = self((img1, img2), training=False)
        loss = self.loss_function(label, output)
        self.metrics[0].update_state(label, output)
        self.metrics[1].update_state(loss)

        return {m.name : m.result() for m in self.metrics}


def training_loop(model, train_ds, val_ds, start_epoch,
                  epochs, train_summary_writer, 
                  val_summary_writer, save_path):

    # 1. iterate over epochs
    for e in range(start_epoch, epochs):

        # 2. train steps on all batches in the training data
        for data in tqdm.tqdm(train_ds, position=0, leave=True):
            metrics = model.train_step(data)

        # 3. log and print training metrics

        with train_summary_writer.as_default():
            # for scalar metrics:
            for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=e)
            # alternatively, log metrics individually (allows for non-scalar metrics such as tf.keras.metrics.MeanTensor)
            # e.g. tf.summary.image(name="mean_activation_layer3", data = metrics["mean_activation_layer3"],step=e)
        
        #print the metrics
        print([f"{key}: {value.numpy()}" for (key, value) in metrics.items()])
        
        # 4. reset metric objects
        model.reset_metrics()


        # 5. evaluate on validation data
        for data in val_ds:
            metrics = model.test_step(data)
        

        # 6. log validation metrics

        with val_summary_writer.as_default():
            # for scalar metrics:
            for metric in model.metrics:
                    tf.summary.scalar(f"{metric.name}", metric.result(), step=e)
            # alternatively, log metrics individually (allows for non-scalar metrics such as tf.keras.metrics.MeanTensor)
            # e.g. tf.summary.image(name="mean_activation_layer3", data = metrics["mean_activation_layer3"],step=e)
            
        print([f"val_{key}: {value.numpy()}" for (key, value) in metrics.items()])
        # 7. reset metric objects
        model.reset_metrics()
        
    # 8. save model weights if save_path is given
    if save_path:
        model.save_weights(save_path)
    


### Summary Writers 
def create_summary_writers(config_name="RUN"):
    
    # Define where to save the logs
    # along with this, you may want to save a config file with the same name so you know what the hyperparameters were used
    # alternatively make a copy of the code that is used for later reference
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_path = f"Homework 04/logs/{config_name}/{current_time}/train"
    val_log_path = f"Homework 04/logs/{config_name}/{current_time}/val"

    # log writer for training metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_path)

    # log writer for validation metrics
    val_summary_writer = tf.summary.create_file_writer(val_log_path)
    
    return train_summary_writer, val_summary_writer

train_summary_writer, val_summary_writer = create_summary_writers(config_name="SGD")


# 1. instantiate model
model = TwinMNISTModel()

# 2. choose a path to save the weights
save_path = "Homework 04/trained_model_SGD"

# 2. pass arguments to training loop function

training_loop(model=model,
    train_ds=train_ds,
    val_ds=val_ds,
    start_epoch=0,
    epochs=10,
    train_summary_writer=train_summary_writer,
    val_summary_writer=val_summary_writer,
    save_path=save_path)


# load the model:
fresh_model = TwinMNISTModel()

# build the model's parameters by calling it on input
for img1,img2,label in train_ds:
    fresh_model((img1,img2))
    break

# load the saved weights
#model = fresh_model.load_weights(save_path)
