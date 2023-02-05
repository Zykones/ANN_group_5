import tensorflow as tf 

class SkipGram(tf.keras.models.Model):
    """ A SkipGram model to create word embeddings. """
    
    def __init__(self, vocabulary_size : int, embedding_size : int = 64):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.num_sampled = input_shape[0]
        self.embedding = self.add_weight(
            name = "embedding",
            shape=(self.vocabulary_size,self.embedding_size),
            initializer = "uniform",
            trainable = True
        )
        self.score_weight = self.add_weight(
            name='score',
            shape=(self.vocabulary_size,self.embedding_size),
            initializer='uniform',
            trainable=True
        )
        self.score_bias = self.add_weight(
            name='score_bias',
            shape=(self.vocabulary_size, ),
            initializer='zero',
            trainable=True
        )
        super().build(input_shape)

    def call(self, input, training = None): # not done
        embedding = tf.nn.embedding_lookup(self.embedding, input)
        return embedding

    def train_step(self,inputs):
        t,c = inputs

        with tf.GradientTape() as tape:
            target_embedding = self(t,True)
            loss = tf.reduce_mean(tf.nn.nce_loss(
                weights=self.score_weight,
                biases=self.score_bias,
                labels=c,
                inputs=target_embedding,
                num_sampled = self.num_sampled,
                num_classes=self.vocabulary_size
            ))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.trainable_variables))
        
        return {'loss': loss}
        