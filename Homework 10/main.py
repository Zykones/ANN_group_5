import tensorflow as tf

from data_processing import get_preprocessed_data
from model import SkipGram

# loading data
file_name = "bible.txt"
file_path = f"data/{file_name}"

VOCABULARY = 10000
WINDOW = 2 # because window is i - 2 and i + 2 then = 5
TRAIN_PART = 0.8 # partition of the data to be training data
EMBEDDING = 64 # size of the embedding

#(train_ds, test_ds), tokenizer = get_preprocessed_data(file_path,VOCABULARY,WINDOW,TRAIN_PART)

model = SkipGram(VOCABULARY,EMBEDDING)
model.compile(optimizer='adam')