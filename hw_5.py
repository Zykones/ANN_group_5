import tensorflow as tf
import tensorflow_datasets as tfds

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

