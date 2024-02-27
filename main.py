import tensorflow as tf

import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


data_dir = pathlib.Path('mri_tumor_classification/Dataset') # add "mri_tumor_classification/" before Dataset to run on windows
train_dir = (data_dir / 'Training')
test_dir = data_dir / 'Testing'


glioma_imgs = list(train_dir.glob('glioma/*'))
meningioma_imgs = list(train_dir.glob('meningioma/*'))
notumor_imgs = list(train_dir.glob('notumor/*'))
pituitary_imgs = list(train_dir.glob('pituitary/*'))

batch_size = 64
img_height = 256
img_width = 256

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=000,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    seed=000,
    color_mode='grayscale',
    image_size=(img_height, img_width),
     batch_size=batch_size)

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# print(np.min(first_image), np.max(first_image))

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

plt.figure(figsize=(10, 10))
for images, labels in val_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy())
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=20
)