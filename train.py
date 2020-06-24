import tensorflow.keras as keras
import pandas as pd
import numpy as np
import time
from datetime import date
from sklearn.preprocessing import StandardScaler
import pickle


# Load data
with open("processed_stonk_data", "rb") as f:
    X_data, y_data = pickle.load(f)

# Split data
X_train, y_train = X_data[:-10], y_data[:-10]
X_test, y_test = X_data[-10:], y_data[-10:]


# Build model and run
model = keras.models.Sequential([
    keras.layers.LSTM(40, activation="elu", kernel_initializer="he_normal", return_sequences=True, input_shape=X_train.shape[1:]),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(20, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])

opt = keras.optimizers.Adam(lr=1e-3, decay=1e-5)

print(model.summary())

model.compile(optimizer=opt, loss="mse")

history = model.fit(X_train, y_train, epochs=500)

model.evaluate(X_test, y_test)
