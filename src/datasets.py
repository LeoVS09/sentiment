import tensorflow_datasets as tfds
import tensorflow as tf
from .libs import params
import pandas as pd
import os
from progressbar import ProgressBar, UnknownLength

# Loaded from http://help.sentiment140.com/for-students
# kaggle copy https://www.kaggle.com/kazanova/sentiment140

datasets_folder = './data'

train_dataset_path = os.path.join(datasets_folder, 'training.1600000.processed.noemoticon.csv')
test_dataset_path = os.path.join(datasets_folder, 'testdata.manual.2009.06.14.csv')

# TODO: split initial train dataset into smallest development dataset (with same distribution of classes)
#  for allow train locally and build smallest vocabulary

LABEL_COLUMN = 'target'
TEXT_COLUMN = 'text'
BATCH_SIZE = params['input']['batch_size']
COLUMNS = ["target", "id", "date", "flag", "user", "text"]

# Train dataset must contain labels in format: 0 = negative, 2 = neutral, 4 = positive
# but actually conain only "0" and "4", can be string or number
# test dataset also contains "2"
def normalize_label(label):
    if label == "0" or label == 0:
        return 0

    if label == "2" or label == 2:
        return 0.5

    return 1

def get_dataset_iterator(file_path, display_progress=False, chunk_size=10, chunk_count=UnknownLength):
    print('Start reading dataset from', file_path)

    bar = None
    if display_progress:
        bar = ProgressBar(max_value=chunk_count, max_error=False).start()
    
    for i, chunk in enumerate(pd.read_csv(file_path, encoding = "ISO-8859-1", names=COLUMNS, chunksize=chunk_size)):
        if bar != None:
            bar.update(i)

        for item in chunk.index:
            text = chunk[TEXT_COLUMN][item]
            label = chunk[LABEL_COLUMN][item]
            
            label = normalize_label(label)
            
            yield (text, label)

    if bar != None:
        bar.finish()

TRAINING_CHUNK_SIZE = 1000 # Read thousand records at once
TRAINING_RECORDS_COUNT = 1600000
TRAINING_CHUNKS_COUNT = TRAINING_RECORDS_COUNT // TRAINING_CHUNK_SIZE

def get_train_dataset_iterator(display_progress=False):
    return get_dataset_iterator(
        train_dataset_path, 
        display_progress=display_progress, 
        chunk_size=TRAINING_CHUNK_SIZE,
        chunk_count=TRAINING_CHUNKS_COUNT
    )

def get_test_dataset_iterator(display_progress=False):
    return get_dataset_iterator(test_dataset_path, display_progress=display_progress, chunk_count=500)

def get_dataset(generator):
    return tf.data.Dataset.from_generator(
        generator, 
        (tf.string, tf.int64), 
        ((), ())
    )

def get_train_dataset(display_progress=False):
    generator = lambda: get_train_dataset_iterator(display_progress=display_progress)
    return get_dataset(generator)

def get_test_dataset(display_progress=False):
    generator = lambda: get_test_dataset_iterator(display_progress=display_progress)
    return get_dataset(generator) 

def download(display_train_progress=False, display_test_progress=False):
    train_dataset = get_train_dataset(display_progress=display_train_progress)
    test_dataset = get_test_dataset(display_progress=display_test_progress)

    return train_dataset, test_dataset

# Will print dataset sizes
# Not use it in production, 
# size of dataset can be computed only by transformation to list
def print_dataset_sizes(train_data, test_data):
    print(
        '\nLoaded dataset',
        '\ntrain size:', len(list(train_data)),
        '\ntest size:', len(list(test_data)), '\n'
    )

def get_item(dataset, index):
    (text, label) = list(dataset.take(index+1).as_numpy_iterator())[index]
    return text, label