from __future__ import print_function
#from __future__ import division

import sys
import os
import copy
import time
import random
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras import optimizers

import numpy as np

import csv

# =============== Parameters ==========================

optlist = [ 'rmsprop', 'sgd', 'adam', 'nadam', 'adamax', 'adadelta', 'adagrad' ]
actlist = [ 'relu', 'tanh', 'sigmoid', 'softplus' ]
totalEpochs = 1000
fitTime = 600 # seconds
tries = 2 # tries for each config. Getting best try as fitness
maxLayers = 2
popsize = 12
ageOfRemoval = 3 # after that many epochs individuum will be removed from population

print("Estimated epoch time: " + str( 0.75 * popsize * fitTime / 60.0) + " minutes")

# =============== Dataset etc. =======================

print("Reading dataset...")

fin = open("cleaned2sigma_train.csv", 'rt')
reader = csv.reader(fin)

data = [d for d in reader]
val_data = data[-2000:]
#data = data[:60000]

fin.close()

X = np.asarray(data, dtype = np.float32)
val_x = np.asarray(val_data, dtype = np.float32)

targets = X[:, [11]]
val_y = val_x[:, [11]]

X = np.delete(X, 11, 1)
val_x = np.delete(val_x, 11, 1)

def createNetByConfigs(cfg):
    wCfg, actCfg, opt, drop = cfg

    model = Sequential()
    model.add(Dense(wCfg[0], activation=actCfg[0], input_dim=11))

    for w, act, d in zip(wCfg[1:], actCfg[1:], drop[1:]):
        if w > 0:
            if d > 0:
                model.add(Dropout(d))
            model.add(Dense(w, activation = act))

    model.compile(
        loss='binary_crossentropy',
        optimizer=opt
    )

    return model

cfg = ( [1021, 140, 1], 
        ['tanh', 'sigmoid', 'sigmoid'], 
        'adamax', 
        [-0.20152076744570152, 0.07689980538208308, 0.7510180629883729]
      )

model = createNetByConfigs(cfg)

for _ in range(1000):
    p = np.random.permutation(len(X))
    history = model.fit(X[p], targets[p],
            #validation_data = (val_x, val_y),
            validation_split = 0.05,
            shuffle = False,
            batch_size=len(X),
            epochs=10,
            verbose=1)
    
    loss = np.asarray(history.history['loss'], dtype = np.float32).mean()
    val_loss = np.asarray(history.history['val_loss'], dtype = np.float32).mean()

    if val_loss < 0.55 and loss * val_loss < 0.32:
        model.save("models/" + str(val_loss) + "_" + str(loss) + ".h5")