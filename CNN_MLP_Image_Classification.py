# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Reshape
from tensorflow.keras.utils import to_categorical
import pandas as pd

# Check if TensorFlow is using the GPU
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please ensure that you have enabled GPU acceleration in Colab's runtime settings.")

# Load the data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

x_train = train_images
y_train = train_labels
x_test = test_images
y_test = test_labels

# Flatten the images
x_train_mlp = x_train.reshape(-1, 28*28)
x_test_mlp = x_test.reshape(-1, 28*28)

# Normalize pixel values (between 0 and 1)
x_train_mlp = x_train_mlp.astype('float32') / 255
x_test_mlp = x_test_mlp.astype('float32') / 255

# One-hot encode labels
y_train_mlp = to_categorical(y_train)
y_test_mlp = to_categorical(y_test)

print("MLP Training set shape:", x_train_mlp.shape)
print("MLP Test set shape:", x_test_mlp.shape)

# Reshape the training and test images to include channel dimension (28x28x1)
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# Normalize pixel values (between 0 and 1)
x_train_cnn = x_train_cnn.astype('float32') / 255
x_test_cnn = x_test_cnn.astype('float32') / 255

y_train_cnn = y_train
y_test_cnn = y_test

print("CNN Training set shape:", x_train_cnn.shape)
print("CNN Test set shape:", x_test_cnn.shape)

# Define MLP model architecture
model_mlp = Sequential()
model_mlp.add(Dense(512, input_shape=(784,), activation='relu'))
model_mlp.add(Dense(256, activation='relu'))
model_mlp.add(Dense(10, activation='softmax'))

# Compile MLP model
model_mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train MLP model
mlp_history = model_mlp.fit(x_train_mlp, y_train_mlp, epochs=10, batch_size=128, validation_data=(x_test_mlp, y_test_mlp), verbose=1)

# Define model architecture
model_cnn = Sequential()
model_cnn.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_cnn.add(MaxPool2D((2, 2)))
model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(MaxPool2D((2, 2)))
model_cnn.add(Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(Flatten())
model_cnn.add(Dense(64, activation='relu'))
model_cnn.add(Dense(10, activation='softmax'))

# Compile CNN model
model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_history = model_cnn.fit(x_train_cnn, y_train_cnn, epochs=10, batch_size=128, validation_data=(x_test_cnn, y_test_cnn), verbose=1)

# Evaluate MLP model
test_loss, test_acc = model_mlp.evaluate(x_test_mlp, y_test_mlp)
print(f'Test accuracy: {test_acc:.2f}')

# Evaluate CNN model
test_loss, test_acc = model_cnn.evaluate(x_test_cnn, y_test_cnn)
print(f'Test accuracy: {test_acc:.2f}')

# Extract metrics from the history objects
mlp_train_loss = mlp_history.history['loss']
mlp_val_loss = mlp_history.history['val_loss']
mlp_train_acc = mlp_history.history['accuracy']
mlp_val_acc = mlp_history.history['val_accuracy']

cnn_train_loss = cnn_history.history['loss']
cnn_val_loss = cnn_history.history['val_loss']
cnn_train_acc = cnn_history.history['accuracy']
cnn_val_acc = cnn_history.history['val_accuracy']

# Create a DataFrame
data = {
    'Epoch': range(1, len(mlp_train_loss) + 1),
    'MLP Train Loss': mlp_train_loss,
    'MLP Val Loss': mlp_val_loss,
    'MLP Train Acc': mlp_train_acc,
    'MLP Val Acc': mlp_val_acc,
    'CNN Train Loss': cnn_train_loss,
    'CNN Val Loss': cnn_val_loss,
    'CNN Train Acc': cnn_train_acc,
    'CNN Val Acc': cnn_val_acc
}

df = pd.DataFrame(data)
print(df)