# -*- coding: utf-8 -*-
"""IRIS_NEURALNET.ipynb

#1.BUILD A NEURAL NETWORK FOR IRIS DATA CLASSIFICATION
This document lists the code required for building neural network for iris data

##1.1 IMPORT NECESSARY LIBRARIES
"""

# prompt: import necessary libraries for neural network including keras

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

"""##1.2 IMPORT IRIS DATASET"""

# prompt: IMPORT THE IRIS DATASET

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
# print(iris.DESCR)
# print(X.shape)
# print(y.shape)
# print(X[0])
# print(y[0])
print(X)
print(y)

"""##1.3 SCALE THE INDEPENDENT VARIABLES
Training neural network requires to scale the independent variables. let us apply the scaling for the variables.
"""

# prompt: apply scaling for X variable to feed to neural network model

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X

# prompt: partition the data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
y_test.shape

"""##2. Build the neural network architecture"""

# prompt: build a neural network with 1 hidden layer and 5 nodes, output activation is sigmoid for classification model

# Continue from the provided code

# Define the model
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(4,)),  # Hidden layer with 5 nodes and ReLU activation
    layers.Dense(3, activation='sigmoid')  # Output layer with 3 nodes (for 3 classes) and sigmoid activation
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse categorical crossentropy for integer labels
              metrics=['accuracy'])
#check the input shape
print(X_train.shape)

# Train the model
history = []
history = model.fit(X_train, y_train, epochs=200, batch_size= 32)  # Adjust the number of epochs as needed

loss = history.history['loss']
accuracy = history.history['accuracy']

# Evaluate the model (optional)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# prompt: draw a line chart of accuracy and loss list

import matplotlib.pyplot as plt

# Assuming 'history' object from model training is available

# Plot the accuracy
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot the loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# prompt: do the model prediction for y test

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)
print(y_test)
