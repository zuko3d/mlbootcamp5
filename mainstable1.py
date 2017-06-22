from __future__ import print_function
import time
import random
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras import optimizers

import numpy as np

import csv

print("Reading dataset...")

fin = open("cleaned_train.csv", 'rt')
reader = csv.reader(fin)

data = [d for d in reader]
val_data = data[-100:]
data = data[:400]

fin.close()

X = np.asarray(data, dtype = np.float32)
val_x = np.asarray(val_data, dtype = np.float32)

targets = X[:, [11]]
val_y = val_x[:, [11]]

X = np.delete(X, 11, 1)
val_x = np.delete(val_x, 11, 1)

# X = np.delete(X, 10, 1)
# X = np.delete(X, 9, 1)
# X = np.delete(X, 8, 1)
# X = np.delete(X, 1, 1)

#X = (X - X.min(0)) / X.ptp(0)

print(X)

#Y = Y * 2 - 1

# z = np.copy(targets)
# z = 1 - z

# targets = np.concatenate((targets,z), axis = 1)

# z = np.copy(val_y)
# z = 1 - z

# val_y = np.concatenate((val_y,z), axis = 1)

bestval = 1.0
bestloss = 1.0
best = []
bestl = []

flog = open("log_" + time.strftime("%d_%H_%M_%S") + ".txt", 'w')

# for i1 in range(3):
#     w1 = (9 ** i1) * 3
#     for i2 in range(3):
#         w2 = (8 ** i2) * 1
#         # if w2 > w1:
#         #     continue
#         for i3 in range(3):
#             w3 = (3 ** i3) * 3
#             # if w3 > w2:
#             #     continue
#             for i4 in range(3):
#                 w4 = (3 ** i4) * 1
                
#                 if w1 * w2 + w2 * w3 + w3 * w4 < 200:
#                     continue
                # if w4 > w3:
                #     continue
for _ in range(1000):
    w1 = random.randint(10, 700)
    w2 = random.randint(-500, 700)
    w3 = random.randint(-1100, 700)
    w4 = random.randint(-500, 700)

    act = ['tanh', 'sigmoid', 'relu'][random.randint(0, 2)]
    act1 = ['tanh', 'sigmoid', 'relu'][random.randint(0, 2)]
    act2 = ['tanh', 'sigmoid', 'relu'][random.randint(0, 2)]
    actout = 'sigmoid'
    drop = [True, False][random.randint(0, 1)]
    # for act in ['tanh', 'sigmoid', 'relu']:
    #     for act1 in ['tanh', 'sigmoid', 'relu']:
    #         for act2 in ['tanh', 'sigmoid', 'relu']:
    #             for actout in ['sigmoid']:
    #                 for drop in [False]:

    model = Sequential()
    model.add(Dense(w1, activation=act1, input_dim=11))

    if drop:
        model.add(Dropout(0.5))

    if w2 > 1:
        model.add(Dense(w2, activation=act1))

    if w3 > 1:
        model.add(Dense(w3, activation=act))

    if w4 > 1:
        model.add(Dense(w4, activation=act2))

    model.add(Dense(1, activation=actout))

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.01)
    )

    cfg = [w1, drop, w2, w3, w4, act1, act, act2, actout]

    flog.write(str(cfg))
    print("cfg: ", cfg)

    ts = time.time()

    
    ls = time.time()
    while time.time() - ls < 60:
        model.fit(  X, targets,
                    #validation_data = (val_x, val_y),
                    batch_size=len(X),
                    epochs=10,
                    verbose=0)

    history = model.fit(X, targets,
                validation_data = (val_x, val_y),
                batch_size=len(X),
                epochs=10,
                verbose=0)

    loss = np.asarray(history.history['loss'], dtype = np.float32).mean()
    val_loss = np.asarray(history.history['val_loss'], dtype = np.float32).mean()
    
    if val_loss * loss < 0.36:
        print("loss: ", loss)
        print("val_loss: ", val_loss)

        if bestval > val_loss:
            bestval = val_loss
            best = cfg
            print("new val record! <=================")

        if bestloss > loss:
            bestloss = loss
            bestl = cfg
            print("new loss record! <=================")

        model.save("models/" + str(loss) + "_" + str(val_loss) + "_model.h5")

    flog.write(str([loss, val_loss]))
    flog.write('\n')
    flog.flush()

    del model
print("best val: ", bestval)
print("best: ", best)

print("best loss: ", bestloss)
print("bestl: ", bestl)

flog.close()