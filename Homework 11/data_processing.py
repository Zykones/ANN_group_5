import tensorflow as tf
import re
import sentencepiece as sp
import tensorflow_text as tf_txt

def load_data(path):
    with open(path) as f:
        data = f.read()
    return data

def data_preprocess(data, batch_size :int = 64):
    """ creates a data pipeline to preprocess the tensorflow datasets mnst dataset
    
    Parameters: 
        data (tf.data.Dataset) = the dataset to preprocess
        batch_size (int) = size of the batches
    """
    #cache shuffle, batch, prefetch
    data = data.cache()
    data = data.shuffle(3000)
    data = data.batch(batch_size)     
    data = data.prefetch(tf.data.AUTOTUNE)
    return data

def prepare_data(data,file_path = None):
    """ prepares the data and saved it at file_path if file_path is not None"""
    #prepare text and save
    prepared_data = re.sub(r'[\n]+','\n',data.lower())
    prepared_data = re.sub(r'[^\sa-zA-Z.!?]+','',prepared_data)
    #prepared_data = " ".join(prepared_data.split())   

    if(file_path):
        text_file = open(file_path, "w")
        text_file.write(prepared_data)
        text_file.close()

    return prepared_data
    
def train_tokenizer(data_path,vocabulary_size,model_prefix):
    """ train a SentencePieceTrainer on the data in data_path """
    sp.SentencePieceTrainer.train(input = data_path, model_prefix=model_prefix, model_type="unigram", vocab_size=vocabulary_size)

def load_tokenizer(model_prefix):
    # deserialize the trained model file to load it in the correct format
    trained_tokenizer_model = tf.io.gfile.GFile(model_prefix + '.model', "rb").read()

    # load the model as a tokenizer that can be used inside a tensorflow model
    tokenizer = tf_txt.SentencepieceTokenizer(
        model=trained_tokenizer_model, out_type=tf.int32, nbest_size=-1, alpha=1, reverse=False,
        add_bos=False, add_eos=False, return_nbest=False, name=None
    )
    return tokenizer

def prepare_everything(original_file_path,prepared_file_path,model_prefix,vocabulary_size):
    """ returns original data, prepared data, tokenizer 
    but also creates prepared_data and tokenizer newly"""

    data = load_data(original_file_path)
    prepared_data = prepare_data(data,prepared_file_path)
    train_tokenizer(prepared_file_path,vocabulary_size,model_prefix)
    return data, prepared_data, load_tokenizer(model_prefix)

def load_everything(original_file_path,prepared_file_path,model_prefix):
    return load_data(original_file_path), load_data(prepared_file_path),load_tokenizer(model_prefix)

def create_dataset(prepared_data, tokenizer, window_size,batch_size):
    """ takes a prepared text string as data and tokenizes it, then creates sliding windows, 
    makes a tf Dataset out of it and finally shuffle, batch, prefetch"""

    tokens = tokenizer.tokenize(prepared_data)
    windows = tf_txt.sliding_window(tokens, window_size + 1, axis=0)
    ds = tf.data.Dataset.from_tensor_slices(windows)
    ds = data_preprocess(ds,batch_size)
    return ds
