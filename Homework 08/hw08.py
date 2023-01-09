import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os

#Line only needed to run Code on my Surface, can be commented out
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

class MyCNNNormalizationLayer(tf.keras.layers.Layer):
    """ a layer for a CNN with kernel size 3 and ReLu as the activation function """

    def __init__(self,filters,normalization=False, reg = None):
        """ Constructor
        
        Parameters: 
            filters (int) = how many filters the Conv2D layer will have
            normalization (boolean) = whether the output of the layer should be normalized 
        """
        super(MyCNNNormalizationLayer, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding='same', kernel_regularizer = reg)
        self.norm_layer = tf.keras.layers.BatchNormalization() if normalization else None
        self.activation = tf.keras.layers.Activation("relu")

    @tf.function
    def call(self,x,training=None):
        """ forward propagation """

        x = self.conv_layer(x)
        if self.norm_layer:
            x = self.norm_layer(x,training)
        x = self.activation(x)

        return x

class MyCNNBlock(tf.keras.layers.Layer):
    """ a block for a CNN having several convoluted layers with filters and kernel size 3 and ReLu as the activation function """

    def __init__(self,layers,filters,global_pool = False,mode = False,normalization = False, reg = None, dropout_layer = None):
        """ Constructor 
        
        Parameters: 
            layers (int) = how many Conv2D you want
            filters (int) = how many filters the Conv2D layers should have
            global_pool (boolean) = global average pooling at the end if True else MaxPooling2D
            denseNet (boolean) = whether we want to implement a denseNet (creates a concatenate layer if True)
        """

        super(MyCNNBlock, self).__init__()
        self.dropout_layer = dropout_layer
        self.conv_layers =  [MyCNNNormalizationLayer(filters,normalization, reg) for _ in range(layers)]
        self.mode = mode
        switch_mode = {"dense":tf.keras.layers.Concatenate(axis=-1), "res": tf.keras.layers.Add(),}
        self.extra_layer = None if mode == None else switch_mode.get(mode,f"{mode} is not a valid mode for MyCNN. Choose from 'dense' or 'res'.")
        self.pool = tf.keras.layers.GlobalAvgPool2D() if global_pool else tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

    @tf.function
    def call(self,input,training=None):
        """ forward propagation of this block """
        x = input
        for i, layer in enumerate(self.conv_layers):
            x = layer(x,training)
            if(i==0 and self.mode == "res"): # for resnet add output of first layer to final output, not input of first layer
                input = x
            if self.dropout_layer:
                x = self.dropout_layer(x, training)
        if(self.extra_layer is not None):
            x = self.extra_layer([input,x])

        x = self.pool(x)
        return x

class MyCNN(tf.keras.Model):
    """ an CNN created to train on Cifar-10 """
    
    def __init__(self, optimizer, output_units : int = 10, filter_start = 24, mode = None,normalization = False,dropout_rate = None, regularizer = None):
        """ Constructor 
        
        Parameters: 
            optimizer = the optimizer to use for training
            output_units (int) = the number of wanted output units
            filter_start (int) = filters for the first CNN Block
            mode (String) = whether to implement a DenseNet "dense" ore a ResNet "res"
            normalization (boolean) = whether to have normalization layers
            dropout_rate (0<= int <1) = rate of dropout for after input and after dense
            regularizer (0<= int <1) = rate for l1 and l2 regularizer
        """

        super(MyCNN, self).__init__()

        self.reg = regularizer
        self.dropout_rate = dropout_rate
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate) if self.dropout_rate else None

        self.block1 = MyCNNBlock(layers = 2,filters = filter_start,mode = mode,normalization = normalization, reg = self.reg, dropout_layer = self.dropout_layer)
        self.block2 = MyCNNBlock(layers = 2,filters = filter_start*2,mode = mode,normalization = normalization, reg = self.reg, dropout_layer = self.dropout_layer)

        self.flatten = tf.keras.layers.Flatten()
        self.out = tf.keras.layers.Dense(output_units, activation=tf.nn.relu)

    @tf.function
    def call(self, x, training = False):
        """ forward propagation of the ANN """
        
        x = self.block1(x,training = training)
        x = self.block2(x,training = training)

        x = self.flatten(x)
        x = self.out(x,training = training)
        return x

class MyDecoder(tf.keras.Model):

    def __init__(self,filter_start = 24,):

        super(MyDecoder,self).__init__()

        self.dense1 = tf.keras.layers.Dense(7*7*filter_start*2)
        self.reshape = tf.keras.layers.Reshape((7,7,filter_start*2)) # batch size nicht !

        self.trans1 = tf.keras.layers.Conv2DTranspose(filter_start,kernel_size=3,strides = 2, padding="same") # strides = 2 doubles the size of the image
        self.trans2 = tf.keras.layers.Conv2DTranspose(1,kernel_size=3,strides = 2, padding="same",activation=tf.nn.sigmoid)

    @tf.function
    def call(self, x, training = False):
        """ forward propagation of the Decoder"""
        
        x = self.dense1(x)
        x = self.reshape(x)
        x = self.trans1(x)
        x = self.trans2(x)
        return x

class MyAutoencoder(tf.keras.Model): 

    def __init__(self,encoder,decoder):

        super(MyAutoencoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.metrices = [tf.keras.metrics.Mean(name = "loss"), tf.keras.metrics.Accuracy(name = "accuracy")]

    def reset_metrics(self):
        """ resets all the metrices that are observed during training and testing """
        for m in self.metrics:
            m.reset_states()

    @tf.function
    def call(self, x, training = False):
        """ forward propagation of the Autoencoder"""
        
        x = self.encoder(x,training = training)
        x = self.decoder(x,training = training)

        return x

    @tf.function
    def train_step(self, data):  
        """
        Standard train_step method, assuming we use model.compile(optimizer, loss, ...)
        """
        
        image, target = data
        with tf.GradientTape() as tape:
            output = self(image, training=True)
            loss = self.compiled_loss(target, output, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(target, output)
        
        return {m.name : m.result() for m in self.metrics}
    
    @tf.function
    def test_step(self, data):       
        """
        Standard test_step method, assuming we use model.compile(optimizer, loss, ...)
        """
        
        image, target = data
        output = self(image, training=False)
        loss = self.compiled_loss(target, output, regularization_losses=self.losses)
                
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(target, output)
        
        return {m.name : m.result() for m in self.metrics}

def load_data(info : bool = False):
    """ loads the mnst dataset from tensorflow datasets 
    
    Parameters: 
        info (bool) = wether you want some info to be displayed and also additionaly returned
    """

    ( train_ds , test_ds ) , ds_info = tfds.load("mnist", split =[ "train", "test"], as_supervised = True , with_info = True )

    if(info):
        print(ds_info)
        tfds.show_examples(train_ds, ds_info)
        return (train_ds, test_ds) , ds_info

    return (train_ds, test_ds) , ds_info

def data_preprocess(data, batch_size = 64,noisy = 0.1,targets = False):
    """ creates a data pipeline to preprocess the tensorflow datasets mnst dataset
    
    Parameters: 
        data (tensorflow.data.Dataset) = the dataset to preprocess
        noisy (float) = standard deviation of the noise to add around 0
    """

    # cast to float
    data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # normalize the image values
    data = data.map(lambda img, target: ((img/255), target)) # create in between 0 and 1

    # create the specific data we want: 

    # add a color channel dimension (the color channel dimension already exists in mnist)
    # data = data.map(lambda img, target: ( tf.expand_dims(img,axis=-1), target ) )

    if targets:
        targets_data = data.map(lambda img, target: target)

    # remove the targets, add noise instead
    data = data.map(lambda img, target: (tf.random.normal(shape=img.shape,mean=0,stddev=noisy), img) )


    # add noise to images and save in new dataset
    data = data.map(lambda noise, img: (tf.add(noise,img),img))
    # keep image in the right area
    data = data.map(lambda noise, img: (tf.clip_by_value(noise,clip_value_min=-1,clip_value_max=1),img))

    if targets: 
        data = tf.data.Dataset.zip((data,targets_data))
        data = data.map(lambda images, targets: (images[0],images[1],targets))
    
    #cache shuffle, batch, prefetch
    data = data.cache()
    data = data.shuffle(3000)
    data = data.batch(batch_size)
        
    data = data.prefetch(tf.data.AUTOTUNE)
    return data


if __name__ == "__main__":

    config = "run3"
    epochs = 10 
    batch_size = 64
    noise_std = 0.2
    embedding = 10

    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    loss = tf.losses.MeanSquaredError()

    # get and prepare data and model

    (training_data, o_val_data ) , ds_info = load_data(False) # True
    #training_data = training_data.take(10) # 
    training_data = data_preprocess(training_data, batch_size = batch_size,noisy = noise_std)
    val_data = data_preprocess(o_val_data, batch_size = batch_size, noisy = noise_std)

    encoder = MyCNN(optimizer,embedding,filter_start = 24, regularizer=tf.keras.regularizers.L2(0.001))
    decoder = MyDecoder(24)

    model = MyAutoencoder(encoder,decoder)

    # compile and fit

    model.compile(optimizer = optimizer, loss=loss)

    #logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{config}")

    history = model.fit(training_data, 
                        validation_data = val_data,
                        epochs=epochs, 
                        batch_size=batch_size
                        #callbacks=[logging_callback]
                        )

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(labels=["training","validation"])
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    #plt.savefig(f"Plots/{config}.png")
    plt.show()


