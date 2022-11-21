import tensorflow_datasets as tfds
import tensorflow as tf
import numpy

(train_ds, test_ds), ds_info = tfds.load('mnist', split =['train', 'test'], as_supervised=True, with_info=True)

def prepare_data(mnist, task):
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

    #pair 2 dataset elements (images)
    a = tf.data.Dataset.range(1, 4)  # richtiges dataset eingeben
    b = tf.data.Dataset.range(4, 7)  # dataset!
    ds = tf.data.Dataset.zip((a, b))

    #do calculation
    if task == "regression":
        mse = tf.keras.losses.MeanSquaredError()
        target = mse(y_true, y_pred).numpy() #arguments anpassen
        pass
    elif task == "classification":
        #pair 2 dataset elements
        bce = tf.keras.losses.BinaryCrossentropy()
        target = bce(y_true, y_pred).numpy()
        pass

    #zip target with inputs
    triplet = tf.data.Dataset.zip((ds, target)) # is das n triplet oder ein tuple und ein enzelnes?

    #shuffle, batch, prefetch
    mnist = mnist.shuffle(10000) 
    mnist = mnist.batch(32)
    mnist = mnist.prefetch(20)

    return mnist

prepared_train_ds = prepare_data(train_ds)
prepared_test_ds = prepare_data(test_ds)