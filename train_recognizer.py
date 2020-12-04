import keras
import matplotlib

#
from keras.layers.pooling import MaxPool2D, GlobalMaxPool2D

matplotlib.use("Agg")
# import the necessary packages
#
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU, ReLU
from tensorflow.keras.layers import Dense, LSTM
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras import preprocessing as pc
from keras import layers
from keras.layers import TimeDistributed
from os import path
import numpy as np

# construct the argument parse and parse the arguments

NUM_CLASSES = 10

BASE_PATH = "./data"
# define the batch size
BATCH_SIZE = 64
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])
img_height = 128
img_width = 64
data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)
train_ds = pc.image_dataset_from_directory(BASE_PATH + "/bbox_train",
                                           validation_split=0.2,
                                           label_mode="categorical",
                                           subset="training",
                                           seed=123,
                                           shuffle=False,
                                           image_size=(256, 256),
                                           batch_size=BATCH_SIZE)
val_ds = pc.image_dataset_from_directory(
    BASE_PATH + "/bbox_test",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=BATCH_SIZE)
class_names = train_ds.class_names
print(class_names)

normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))

print(len(normalized_ds))
# extract features from each photo in the directory


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (3, 0.001),
    (6, 0.001),
    (9, 0.0008),
    # (12, 0.001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


# Model Base
# data_augmentation = Sequential(
#     [
#         layers.experimental.preprocessing.RandomFlip("horizontal",
#                                                      input_shape=(img_height,
#                                                                   img_width,
#                                                                   3)),
#         layers.experimental.preprocessing.RandomRotation(0.1),
#         layers.experimental.preprocessing.RandomZoom(0.1),
#     ]
# )
print("[INFO] compiling model...")
# model = Sequential()
print(train_ds)


def build_convnet(shape=(256, 256, 3)):
    # momentum = .9
    model =Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=shape,
                     padding='same', activation='relu'))
    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPool2D())

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())

    # model.add(MaxPool2D(pool_size=(2, 2)))

    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(BatchNormalization())

    # model.add(MaxPool2D(pool_size=(2, 2)))
    #
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(BatchNormalization())

    # flatten...
    model.add(Flatten())
    return model


def action_model(shape=(256, 256, 3), num_class=10):
    # Create our convnet with (112, 112, 3) input shape
    convnet = build_convnet(shape[:])

    # then create our final model
    model = Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add((convnet))
    # here, you can also use GRU or LSTM
    # model.add(LSTM(64))
    # and finally, we make a decision network
    # model.add(Dense(512, activation='relu'))
    # model.add(Dropout(.5))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
    return model
# Block 7: Softmax classifier
# some global params
SIZE = (256, 256)
CHANNELS = 3
# NBFRAME = 1
BS = 8
INSHAPE= SIZE + (CHANNELS,) # (5, 112, 112, 3)
model = action_model(INSHAPE, NUM_CLASSES)
optimizer = Adam(0.001)
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)
EPOCHS=5
# create a "chkp" directory before to run that
# because ModelCheckpoint will write models inside
callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
]
history = model.fit_generator(
    train_ds,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callbacks
)
# model.summary()

# callbacks = [
#     # EpochCheckpoint(args["checkpoints"], every=15, startAt=args["start_epoch"]),
#     #          TrainingMonitor(figPath, jsonPath=jsonPath, startAt=args["start_epoch"]),
#     LearningRateScheduler(lr_schedule)]
# history = model.fit(
#     train_ds,
#     # steps_per_epoch=len(train_ds) // BATCH_SIZE,
#     validation_data=val_ds,
#     # validation_steps=len(val_ds) // BATCH_SIZE,
#     epochs=10,
#     # max_queue_size=BATCH_SIZE * 2,
#     callbacks=callbacks, verbose=1)

epochs = 10
# Plot
import matplotlib as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
