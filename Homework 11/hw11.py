import data_processing as dp
import tensorflow as tf
from model import MyModel
from training_loop import training_loop
import os


# variables from prepare, do not change, or otherwise prepare newly
original_file_path = r"data\bible.txt"
prepared_file_path = r"data\prepared_bible.txt"
model_prefix = 'tokenizer_model'
# Define where to save the log and model
config_name = "bible"
train_summary_writer = tf.summary.create_file_writer(f"logs\\{config_name}\\train")
VOCABULARY_SIZE = 2000 # 2000 - 7000
WINDOW_SIZE = 32 # 32 - 256
BATCH_SIZE = 64
EMBEDDING_DIM = 100 # 64 - 256
NUM_HEADS = 2 # 2-4
FIRST_UNIT = 64 # 32-256

starting_prompt = "first book "
EPOCHS_start = 0 # only needed if you want to continue training
EPOCHS_end = 275 # 100 - 600
TEST_OUTPUT_LENGTH = 30
TOP_K = 20

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# important: tokenizer, optimizer, loss_function, VOCABULARY_SIZE, WINDOW_SIZE, EMBEDDING_DIM, NUM_HEADS, FIRST_UNIT
# create file of model, add all the important data to it
with open(f"data\\{config_name}.txt", "a") as f:
    f.write(f"CONFIG: {config_name}\nEPOCHS: {EPOCHS_start} - {EPOCHS_end}\ndata: {prepared_file_path}\n\nvocabulary size: {VOCABULARY_SIZE}\nwindow size: {WINDOW_SIZE}\nembedding dim: {EMBEDDING_DIM}\n num heads: {NUM_HEADS}\nfirst unit: {FIRST_UNIT}\n\n")

# variables for the model
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits = True)

# Data creation
#**************

# prepare if you want to create a new tokenizer and a new prepared data file
data, prepared_data, tokenizer = dp.prepare_everything(original_file_path,prepared_file_path,model_prefix,VOCABULARY_SIZE)

# if you only want to create a new tokenizer first, use loading afterwards
# dp.train_tokenizer(prepared_file_path,VOCABULARY_SIZE,model_prefix)

# load everything if already prepared
#data, prepared_data, tokenizer = dp.load_everything(original_file_path,prepared_file_path,model_prefix)

dataset = dp.create_dataset(prepared_data,tokenizer,WINDOW_SIZE, BATCH_SIZE)

# Model and training
#*******************

model = MyModel(tokenizer, optimizer, loss_function, VOCABULARY_SIZE, WINDOW_SIZE, EMBEDDING_DIM, NUM_HEADS, FIRST_UNIT)
#model.generate_text("hello dear,", 1, 2)
#model.load_weights(f'model/{config_name}')

training_loop(model,dataset,EPOCHS_start, EPOCHS_end,starting_prompt, TEST_OUTPUT_LENGTH, TOP_K, train_summary_writer,config_name)
