import tensorflow as tf
from .libs import params, prepare, save, checkpoints
from .normalize import datasets
from .model import build_model
from tensorflow.keras.callbacks import CSVLogger

prepare(tf)

BUFFER_SIZE = params['train']['buffer_size'] 
BATCH_SIZE = params['input']['batch_size']
EPOCHS = params['train']['epochs']

metrics_file='metrics/training.csv'

print('Loading and normalizing datasets')
training, testing, vocab_size = datasets()
# Dataset data is array of tensors
# if symplify array of tuples: (text: string, label: int)
# where 0 mean bad, and 1 mean good

# Will split dataset across workers
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

print('Building neural network model')
with strategy.scope():
    model = build_model(vocab_size=vocab_size)

train_batches = training.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
validation_batches = testing.padded_batch(BATCH_SIZE)

print('Start training...')
model.fit(
    train_batches,
    epochs=EPOCHS,
    validation_data=validation_batches,
    callbacks=[
        checkpoints.save_weights(), 
        CSVLogger(metrics_file)
    ]
)

print('Saving for restore in next time') 
save(model)
