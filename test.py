# File used for testing model accuracy
import os
import tensorflow as tf
import tensorflow.keras.layers as network

import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = pathlib.Path('Dataset')
TRAIN_DIR = DATA_DIR / 'Training'
TEST_DIR = DATA_DIR / 'Testing'

BATCH_SIZE = 16
IMG_HEIGHT = 512
IMG_WIDTH = 512

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    color_mode='grayscale',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    color_mode='grayscale',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)


# 
model = tf.keras.Sequential([
    network.experimental.preprocessing.RandomFlip('horizontal'),
    network.Rescaling(1.0/255),

    network.Conv2D(32, 5, activation='relu'),                       
    network.MaxPooling2D(),
    network.Conv2D(32, 5, activation='relu'),
    network.MaxPooling2D(),
    network.Conv2D(32, 5, activation='relu'),
    network.MaxPooling2D(),
    network.Conv2D(32, 5, activation='relu'),
    network.MaxPooling2D(),
    network.Flatten(),

    network.Dense(128, activation='relu'),

    network.Dense(4)
])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=35
)


model.save('Models/m5.h5')

# m1.h5 -> 97.25%
# m2.h5 -> 97.79%
# m3.h5 -> 98.55%
# m4.h5 -> 97.79%
# m5.h5 -> 96.49%