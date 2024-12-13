import tensorflow as tf
from tensorflow import keras
import os

# Dataset Parsing Function
def _parse(example):
    keys_to_features = {
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'label/marks': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_features = tf.io.parse_single_example(example, keys_to_features)

    # Decode the image and resize it
    image_decoded = tf.image.decode_jpeg(parsed_features['image/encoded'], channels=3)
    image_resized = tf.image.resize(image_decoded, [128, 128])  # Resize to INPUT_SHAPE
    image_float = tf.cast(image_resized, tf.float32) / 255.0  # Normalize pixel values

    # Deserialize landmarks and cast to float32
    points = tf.io.parse_tensor(parsed_features['label/marks'], tf.float32)
    points = tf.reshape(points, [-1])  # Ensure shape is consistent
    return image_float, points



def get_parsed_dataset(record_file, batch_size, shuffle=True):
    dataset = tf.data.TFRecordDataset(record_file)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if shuffle:
        dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.map(_parse, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

# Model Definition
class DebugLayer(tf.keras.layers.Layer):
    def __init__(self, message="", **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def call(self, inputs):
        # tf.print(self.message, tf.shape(inputs))
        return inputs

def build_landmark_model(input_shape, output_size):
    preprocess = keras.layers.experimental.preprocessing.Normalization()

    # Define convolutional layers
    conv_1 = keras.layers.Conv2D(32, (3, 3), activation='relu')
    conv_2 = keras.layers.Conv2D(64, (3, 3), activation='relu')
    conv_3 = keras.layers.Conv2D(64, (3, 3), activation='relu')
    conv_4 = keras.layers.Conv2D(64, (3, 3), activation='relu')

    # Define pooling layers
    pool_1 = keras.layers.MaxPool2D((2, 2))
    pool_2 = keras.layers.MaxPool2D((2, 2))
    pool_3 = keras.layers.MaxPool2D((2, 2))
    pool_4 = keras.layers.MaxPool2D((2, 2))

    # Define dense and batch normalization layers
    dense_1 = keras.layers.Dense(1024, activation='relu')
    dense_2 = keras.layers.Dense(output_size, activation=None)

    bn_1 = keras.layers.BatchNormalization()
    bn_2 = keras.layers.BatchNormalization()
    bn_3 = keras.layers.BatchNormalization()
    bn_4 = keras.layers.BatchNormalization()
    bn_5 = keras.layers.BatchNormalization()

    # Flatten layer
    flatten = keras.layers.Flatten()

    # Build the functional model
    inputs = keras.Input(shape=input_shape, name="image_input")
    x = preprocess(inputs)
    x = conv_1(x)
    x = bn_1(x)
    x = pool_1(x)
    x = conv_2(x)
    x = bn_2(x)
    x = pool_2(x)
    x = conv_3(x)
    x = bn_3(x)
    x = pool_3(x)
    x = conv_4(x)
    x = bn_4(x)
    x = pool_4(x)

    # Add a debug layer to check the shape
    x = DebugLayer("Shape after convolution and pooling:")(x)
    x = flatten(x)

    # Dense layers
    x = dense_1(x)
    x = bn_5(x)
    outputs = dense_2(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="landmark")

# Configurations
TRAIN_RECORD = '300w_train.record'
VAL_RECORD = '300w_validation.record'
BATCH_SIZE = 16
EPOCHS = 10
INPUT_SHAPE = (128, 128, 3)
NUM_MARKS = 68
CHECKPOINT_DIR = 'checkpoints'
EXPORT_DIR = 'exported'
LOG_DIR = 'logs'

# Prepare Directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load Datasets
dataset_train = get_parsed_dataset(TRAIN_RECORD, BATCH_SIZE, shuffle=True)
dataset_val = get_parsed_dataset(VAL_RECORD, BATCH_SIZE, shuffle=False)

# Build Model
model = build_landmark_model(INPUT_SHAPE, NUM_MARKS * 2)

# Restore Checkpoint
latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
if latest_checkpoint:
    print(f"Restoring model from {latest_checkpoint}")
    model.load_weights(latest_checkpoint)

# Compile Model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.mean_squared_error)

# Callbacks
callbacks = [
    keras.callbacks.TensorBoard(log_dir=LOG_DIR),
    keras.callbacks.ModelCheckpoint(filepath=os.path.join(CHECKPOINT_DIR, "landmark"),
                                     save_weights_only=True,
                                     save_best_only=True,
                                     monitor='val_loss',
                                     verbose=1)
]

for images, labels in dataset_train.take(1):
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")



# Train Model
model.fit(dataset_train, validation_data=dataset_val, epochs=EPOCHS, callbacks=callbacks)

# Save Model
print(f"Saving model to {EXPORT_DIR}")
model.save(EXPORT_DIR, include_optimizer=False)
