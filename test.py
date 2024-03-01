import tensorflow as tf

import pathlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

data_dir = pathlib.Path('mri_tumor_classification/Dataset')
train_dir = (data_dir / 'Training')
test_dir = data_dir / 'Testing'

batch_size = 32
img_height = 512
img_width = 512

train_ds = tf.keras.utils.image_dataset_from_directory(
    'Dataset//Training',
    seed=000,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

print("Train Size:", train_ds.cardinality().numpy())

val_ds = tf.keras.utils.image_dataset_from_directory(
    'Dataset//Testing',
    seed=000,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size = batch_size)

print("Val Size:", val_ds.cardinality().numpy())

num_classes = len(train_ds.class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

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
    x = train_ds,
    validation_data=val_ds,
    epochs=10
)

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# plt.figure(figsize=(10, 10))
# for images, labels in val_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy())
#         plt.title(class_names[labels[i]])
#         plt.axis("off")
#         plt.colorbar()

# plt.show()

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(val_ds)
print('Predictions len:', len(predictions))

class_names_short = ['G', 'M', 'N', 'P']

correct_count = 0
predictions.flatten()
label_list = []

for images, labels in val_ds:
    label_list.extend(labels.numpy().tolist())


for i in range(len(label_list)):
    if(np.argmax(predictions[i]) == label_list[i]):
        correct_count += 1


print("Correct Count:", correct_count)


j = 0
color = ['red', 'red', 'red', 'red']
same_count = 0
x = 0

# for images, labels in val_ds.take(40):
#     print(class_names[labels[j]])
#     print(class_names[np.argmax(predictions[j])])
#     print('\n')
#     if labels[j] == np.argmax(predictions[j]):
#         same_count += 1
#     # figure, (axis1, axis2) = plt.subplots(1, 2)
#     # plt.title(class_names[labels[j]])
#     # axis1.imshow(images[j].numpy())
#     # #print(class_names[labels[0]])
#     # #print(predictions[0])
#     # index_max = np.argmax(predictions[j])
#     # color[index_max] = 'blue'
#     # axis2.bar(class_names_short, predictions[j], width=0.3, color = color)
#     # plt.xlabel("Prediction")
#     # plt.ylabel("Certainty")
#     # plt.title("Tumor Predictions")
#     # plt.show()
#     j+=1
#     x+=1
#     # for i in range(len(color)):
#     #     color[i] = 'red'

# print("Same count =", same_count, "/", x)

# p=0
# for images, labels in val_ds:
#     print(class_names[labels[p]])
#     print(class_names[np.argmax(predictions[p])])
#     plt.imshow(images[p].numpy())
#     plt.show()
#     p+=1