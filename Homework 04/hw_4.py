import tensorflow_datasets as tfds
import tensorflow as tf
import numpy
import os
import datetime

class TwinMNISTModel(tf.keras.Model):

    # 1. constructor
    def __init__(self):
        super().__init__()
        # inherit functionality from parent class

        # optimizer, loss function and metrics
        self.metrics_list = [tf.keras.metrics.BinaryAccuracy(),
                             tf.keras.metrics.Mean(name="loss")]
        
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        
        # layers to be used
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        
        self.out_layer = tf.keras.layer.Dense(1,activation=tf.nn.sigmoid)
        
        
        
        
    # 2. call method (forward computation)
    #@tf.function
    def call(self, images, training=False):
        img1, img2 = images
        
        img1_x = self.dense1(img1)
        img1_x = self.dense2(img1_x)
        
        img2_x = self.dense1(img2)
        img2_x = self.dense2(img2_x)
        
        combined_x = tf.concat([img1_x, img2_x ], axis=1)
        
        return self.out_layer(combined_x)



    # 3. metrics property
    @property
    def metrics(self):
        return self.metrics_list
        # return a list with all metrics in the model



    # 4. reset all metrics objects
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()



    # 5. train step method
    def train_step(self, data):
        img1, img2, label = data
        
        with tf.GradientTape() as tape:
            output = self((img1, img2), training=True)
            loss = self.loss_function(label, output)
            
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        #something with the metrics object, need to google
        self.metrics_list[0].update_state(label, output)

        # What exactly does the "mean" loss capture in this model? 
        self.metrics_list[1].update_state(loss)
        
        # update the state of the metrics according to loss
        # return a dictionary with metric names as keys and metric results as values

        # Not sure if .numpy() is correct in this context
        return {"Accuracy" : self.metrics_list[0].result().numpy(), "Loss" : self.metrics_list[1].result().numpy()}

    # 6. test_step method
    def test_step(self, data):
        # same as train step (without parameter updates)
        img1, img2, label = data
        output = self((img1, img2), training=False)
        loss = self.loss_function(label, output)
        self.metrics_list[0].update_state(label, output)

        # What exactly does the "mean" of the loss capture in this model? 
        self.metrics_list[1].update_state(loss)

        # Not sure if .numpy() is correct in this context
        return {"Accuracy" : self.metrics_list[0].result().numpy(), "Loss" : self.metrics_list[1].result().numpy()}


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

def training_loop(model, train_ds, test_ds, train_summery_writer, validation_summery_writer, path,  epochs=10):
    for i in range(epochs):
        pass
        # 2. train steps on all batches in the training data
        # 3. log and print training metrics
        with train_summery_writer.as_default():
            pass

        # 4. reset metric objects
        # 5. evaluate on validation data
        # 6. log validation metrics
        with validation_summery_writer.as_default():
            pass
        # 7. reset metric objects
    # 8. save model weights
    
(train_ds, test_ds), ds_info = tfds.load('mnist', split =['train', 'test'], as_supervised=True, with_info=True)

train_ds = prepare_data(train_ds)
test_ds = prepare_data(test_ds)

for img1, img2, label in train_ds.take(1):
    print(img1.shape, img2.shape, label.shape)

### Summary Writers 
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_path = f"logs/train/{current_time}"
val_log_path = f"logs/val/{current_time}"

train_summary_writer = tf.summary.create_file_writer(train_log_path)
val_summary_writer = tf.summary.create_file_writer(val_log_path)