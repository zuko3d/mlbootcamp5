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

print("Reading dataset...")

#fin = open("cleaned2sigma_train.csv", 'rt')
fin = open("test0.5.csv", 'rt')
reader = csv.reader(fin)

data = [d[:-1] for d in reader]

fin.close()

X = np.asarray(data, dtype = np.float32)

model = load_model("models/0.47868_0.527686.h5")

out = model.predict(X)

print(out)

fout = open("ansPure_" + time.strftime("%d_%H_%M_%S") + ".csv", 'wt')
fout01 = open("ans01_" + time.strftime("%d_%H_%M_%S") + ".csv", 'wt')

for x in out:
    fout.write(str(x)[2:-1])
    fout.write('\n')

    if x > 0.5:
        fout01.write('1.0')
    else:
        fout01.write('0.0')
    fout01.write('\n')

fout.close()