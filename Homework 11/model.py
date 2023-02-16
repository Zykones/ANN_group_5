import tensorflow as tf

class MyEmbedder(tf.keras.layers.Layer):

    def __init__(self,vocabulary_size,embedding_dim,sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_layer = tf.keras.layers.Embedding(input_dim = vocabulary_size, output_dim = embedding_dim)
        self.second_embedding_layer = tf.keras.layers.Embedding(input_dim = sequence_length, output_dim = embedding_dim)
        self.add_layer = tf.keras.layers.Add()

    def call(self, input, training = None):
        batch_size = input.shape[0] if input.shape[0] != None else 1
        my_indices = tf.tile(tf.expand_dims(tf.range(0,input.shape[1]),axis=0),multiples = [batch_size,1])

        token_embedding = self.embedding_layer(input)
        indices_embedding = self.second_embedding_layer(my_indices)

        return self.add_layer([token_embedding,indices_embedding])

class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embedding_dim, num_a_heads :int = 2, first_units : int = 32):
        super().__init__()

        self.mha_layer = tf.keras.layers.MultiHeadAttention(num_heads = num_a_heads, key_dim = embedding_dim)

        self.dense1 = tf.keras.layers.Dense(units = first_units, activation = tf.nn.relu) # between 32 and 256 units
        self.dense2 = tf.keras.layers.Dense(units = embedding_dim)

        self.dropout1 = tf.keras.layers.Dropout(rate = 0.1)
        self.dropout2 = tf.keras.layers.Dropout(rate = 0.1)

        self.normalization1 = tf.keras.layers.LayerNormalization(epsilon = 0.000001)
        self.normalization2 = tf.keras.layers.LayerNormalization(epsilon = 0.000001)

        self.add_layer = tf.keras.layers.Add()

    def call(self, input, training = None):
        
        x = self.mha_layer(input, input, training = training, use_causal_mask = True)
        x = self.dropout1(x, training)
        x = self.add_layer([input, x])

        ln_out = self.normalization1(x)
        x = self.dense1(ln_out)
        x = self.dense2(x)
        x = self.dropout2(x, training)
        x = self.add_layer([ln_out, x])

        x = self.normalization2(x)
        return x

class MyModel(tf.keras.Model):

    def __init__(self,tokenizer, optimizer, loss_function, vocabulary_size : int, window_size : int, embedding_dim : int, num_heads : int = 2, first_units : int = 32):
        super().__init__()

        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.vocabulary_size = vocabulary_size
        self.metrics_list = [tf.keras.metrics.Mean(name="loss"), tf.keras.metrics.CategoricalAccuracy(name="accuracy")]

        self.embedding_layer = MyEmbedder(vocabulary_size,embedding_dim,window_size)
        self.transformer_block = TransformerBlock(embedding_dim,num_heads, first_units)
        self.dense = tf.keras.layers.Dense(units = vocabulary_size)

    def reset_metrics(self):
        for metric in self.metrics_list:
            metric.reset_states()

    def call(self, input, training = None):
        x = self.embedding_layer(input, training)
        x = self.transformer_block(x, training)
        x = self.dense(x)
        return x

    # @tf.function
    def train_step(self, data):
        
        x, targets = data[:,:-1], tf.one_hot(data[:,1:],self.vocabulary_size,axis=-1)
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            
            loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics_list[0].update_state(loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics_list[1:]:
            metric.update_state(targets,predictions)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics_list}

    def generate_text(self, prompt, output_length, top_k):
        
        # tokenize the string
        tokenized_string = self.tokenizer.tokenize(prompt)

        # for each token we want to predict
        for _ in range(output_length):

            # let data run through model
            logits = self(tf.expand_dims(tokenized_string,axis=0))

            # take average over input logits results
            logits = tf.reduce_mean(logits,axis=1)

            # only sample from the top k 
            top_k_logits, top_k_indices = tf.math.top_k(input = logits, k = top_k, sorted = True)

            # sample the next token using random categorical
            sample_index = tf.random.categorical(top_k_logits,1)

            # take token
            token = tf.squeeze(top_k_indices,axis=0)[tf.squeeze(sample_index)]

            # concatenate to tokenized_string
            tokenized_string = tf.concat([tokenized_string, tf.expand_dims(token,axis=0)],axis=-1)

        # detokenize the result 
        result = self.tokenizer.detokenize(tokenized_string)
        return result