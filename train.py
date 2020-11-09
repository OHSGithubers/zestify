import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
import pandas as pd
import numpy as np
import time
from datetime import date
from sklearn.preprocessing import StandardScaler
import pickle
from numpy import array
import os

PATH1 = './model.json'
PATH2 = './model.h5'
# Load data
with open("processed_stonk_data", "rb") as f:
    X_data, y_data = pickle.load(f)

# Split data
X_train, y_train = X_data[:-10], y_data[:-10]
X_test, y_test = X_data[-10:], y_data[-10:]

# print("Xtrain0=%s,  Xtrain1=%s Xtrain2=%s " % (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
# print("X0=%s,  X1=%s X2=%s" % (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

if os.path.isfile(PATH1) and os.access(PATH1, os.R_OK)and os.path.isfile(PATH2) and os.access(PATH2, os.R_OK):
    print("Files exist and are readable")

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    model=loaded_model
     # Model checkpoints
    checkpoints = keras.callbacks.ModelCheckpoint("checkpoint_model.h5")

    opt = keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    print(model.summary())

    model.compile(optimizer=opt, loss="mse")

else:
    print("Either the file is missing or not readable")
    print("starting training")



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

    # Model checkpoints
    checkpoints = keras.callbacks.ModelCheckpoint("checkpoint_model.h5")

    opt = keras.optimizers.Adam(lr=1e-3, decay=1e-5)

    print(model.summary())

    model.compile(optimizer=opt, loss="mse")

    history = model.fit(X_train, y_train, epochs=500, callbacks=[checkpoints])

    # serialize model to JSON
    model_json = model.to_json()
    with open("./model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./model.h5")
    print("Saved model to disk")

# evaluate the model
model.evaluate(X_test, y_test)



# print("point 1 X0=%s,  X1=%s X2=%s" % (X_test.shape[0],X_test.shape[1],X_test.shape[2]))

X_test = np.array(X_test)

# print("point 2 X0=%s,  X1=%s X2=%s" % (X_test.shape[0],X_test.shape[1],X_test.shape[2]))

X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1], -1))

# print("X0=%s,  X1=%s X2=%s" % (X_test.shape[0],X_test.shape[1],X_test.shape[2]))

ynew = model.predict(X_test)

print("Predicted=%s " % (ynew))
