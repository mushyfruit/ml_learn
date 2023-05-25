import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

from unet_model import unet

# get the pet dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image(datapoint):
    size = (128, 128)
    input_image = tf.image.resize(datapoint['image'], size)
    input_mask = tf.image.resize(datapoint['segmentation_mask'], size,
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # if tf.random.uniform(()) > 0.5:
    #     input_image = tf.image.flip_left_right(input_image)
    #     input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


# Preprocessing
train_images = dataset['train'].map(load_image,
                                    num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image,
                                  num_parallel_calls=tf.data.AUTOTUNE)

TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 16
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE))
test_batches = test_images.batch(BATCH_SIZE)

# pretrained from VGG16
base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=[128, 128, 3])
for layer in base_model.layers:
    layer.trainable = False

# base_model.summary()
EPOCHS = 3
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits[
                       'test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

model = unet(base_model)
# tf.keras.utils.plot_model(model, show_shapes=True)

for image, mask in train_images.take(1):
    sample_image, sample_mask = image, mask


def show_predictions():
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions(train_images)
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


model_history = model.fit(train_batches,
                          epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback()],
                          use_multiprocessing=True)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

# Show history stats
plt.figure()
plt.plot(model_history.epoch, loss, 'r-', label="Training Loss")
plt.plot(model_history.epoch, val_loss, 'b:', label="Validation Loss")
plt.title("Training + Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend()
plt.show()
