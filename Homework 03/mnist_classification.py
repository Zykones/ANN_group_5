import tensorflow_datasets as tfds
import tensorflow as tf

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


(train_ds , test_ds) , ds_info = tfds . load ("mnist", split =["train", "test"], as_supervised = True , with_info = True)

#print(ds_info)

train_dataset = train_ds.apply(prepare_data)
test_dataset = test_ds.apply(prepare_data)



