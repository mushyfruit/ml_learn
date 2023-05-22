import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

from unet_model import unet

# get the pet dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask


@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (256, 256))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (256, 256))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

# Preprocessing
train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train.cache().shuffle(1000).batch(64).prefetch(buffer_size=tf.data.AUTOTUNE)

# pretrained from VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=[256,256,3])
for layer in base_model.layers:
    layer.trainable = False

# base_model.summary()

model = unet(base_model)
model_history = model.fit(train_dataset, epochs=20)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

# Show history stats
epochs = range(20)
plt.figure()
plt.plot(epochs, loss, 'r-', label="Training Loss")
plt.plot(epochs, val_loss, 'b:', label="Validation Loss")
plt.title("Training + Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend()
plt.show()
