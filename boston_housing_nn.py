APPLY NEURAL NETWORK ON BOSTON HOUSING DATASET.
THIS IS AVAILABLE IN KERAS LIBRARY. SO IMPORT THE DATASET FROM IT

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing


LOAD THE DATA INTO TRAIN AND TEST

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
print(X_train.shape)
print(X_test.shape)


print(X_train)

NORMALIZE THE DATASET BY USING ITS STANDARD DEVIATION AND MEAN.

# Normalize the input data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std #normalization step
X_test = (X_test - mean) / std


CREATE NEURAL NETWORK ARCHITECTURE (1 INPUT LAYER, 2 HIDDEN LAYERS, 1 OUTPUT LAYER)


# Create the model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=13))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1))




NOW COMPILE THE MODEL WITH PARAMETERS AND REQUIRED ACCURACY

model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])


NOW TRAIN THE MODEL

model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

Evaluate the model on the test data

mse, mae = model.evaluate(X_test, y_test, verbose=0)
print('Mean Squared Error:', mse)
print('Mean Absolute Error:', mae)

Do the validation on the Train data.

from matplotlib import pyplot as plt
history = model.fit(X_train, y_train, validation_split = 0.1, epochs=100, batch_size=16, verbose=0)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()