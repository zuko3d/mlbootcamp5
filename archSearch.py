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

fin = open("cleaned_train.csv", 'rt')
reader = csv.reader(fin)

data = [d for d in reader]
val_data = data[-1000:]
data = data[:20000]

fin.close()

X = np.asarray(data, dtype = np.float32)
val_x = np.asarray(val_data, dtype = np.float32)

targets = X[:, [11]]
val_y = val_x[:, [11]]

X = np.delete(X, 11, 1)
val_x = np.delete(val_x, 11, 1)

# =====================================================

def configDiff(cfg1, cfg2):
    wCfg1, actCfg1, opt1, drop1 = cfg1
    wCfg2, actCfg2, opt2, drop2 = cfg2

    dist = 0

    for w1, w2 in zip(wCfg1, wCfg2):
        if w1 != w2:
            dist += 1

    for w1, w2 in zip(actCfg1, actCfg2):
        if w1 != w2:
            dist += 1

    for w1, w2 in zip(drop1, drop2):
            if w1 != w2:
                dist += 1

    if opt1 != opt2:
        dist += 1

    return dist

def calcFitness(cfg, maxtime = fitTime):
    bestval = 10.0
    bestloss = 10.0

    for _ in range(tries):
        model = createNetByConfigs(cfg)

        ls = time.time()
        while time.time() - ls < maxtime / tries:
            model.fit(  X, targets,
                        #validation_data = (val_x, val_y),
                        batch_size=len(X),
                        epochs=40,
                        verbose=0)

        history = model.fit(X, targets,
                    validation_data = (val_x, val_y),
                    batch_size=len(X),
                    epochs=10,
                    verbose=0)

        loss = np.asarray(history.history['loss'], dtype = np.float32).mean()
        val_loss = np.asarray(history.history['val_loss'], dtype = np.float32).mean()

        loss = max(loss, 0.05)

        if loss < bestloss:
            bestloss = loss
        if val_loss < bestval:
            bestval = val_loss

        del model

    return bestval

def crossoverConfigs(cfg1, cfg2, cfg3):
    wCfg1, actCfg1, opt1, drop1 = cfg1
    wCfg2, actCfg2, opt2, drop2 = cfg2
    wCfg3, actCfg3, opt3, drop3 = cfg3

    wCfg = []
    actCfg = []
    drop = []
    
    for w1, w2, w3 in zip(wCfg1, wCfg2, wCfg3):
        wCfg.append(random.choice([w1, w2, w3]))

    for w1, w2, w3 in zip(actCfg1, actCfg2, actCfg3):
        actCfg.append(random.choice([w1, w2, w3]))

    for w1, w2, w3 in zip(drop1, drop2, drop3):
        drop.append(random.choice([w1, w2, w3]))

    opt = random.choice([opt1, opt2, opt3])

    return (wCfg, actCfg, opt, drop)

def mutateConfig(cfg):
    ncfg = copy.deepcopy(cfg)
    wCfg, actCfg, opt, drop = ncfg

    wCfg[random.randint(1, len(wCfg) - 2)] = random.randint(-700, 700)

    actCfg[random.randint(0, len(actCfg) - 1)] = random.choice(actlist)

    drop[random.randint(0, len(drop) - 1)] = random.uniform(-0.5, 1.0)

    opt = random.choice(optlist)

    return (wCfg, actCfg, opt, drop)

def generateConfigForML5(layers = maxLayers):
    wCfg = []
    prevLayer = int(random.gauss(900, 200))
    if prevLayer < 50:
        prevLayer = 50
    wCfg.append(prevLayer)

    prevLayer /= 7
    wCfg.append(prevLayer)

    prevLayer /= 7
    wCfg.append(prevLayer * random.randint(0, 1))

    wCfg.append(1)

    actCfg = []
    drop = []
    for _ in range(len(wCfg)):
        actCfg.append(random.choice(actlist))
        drop.append(random.uniform(-0.5, 1.0))

    actCfg[-1] = 'sigmoid' #!!!!!!!!!!!

    opt = random.choice(optlist)

    return (wCfg, actCfg, opt, drop)

def generateConfigs(layers = maxLayers):
    wCfg = []
    prevLayer = random.randint(700, 1000)
    wCfg.append(prevLayer)

    for i in range(layers):
        prevLayer = random.randint(-prevLayer, prevLayer)
        wCfg.append(prevLayer)
        prevLayer = abs(prevLayer)
        if prevLayer < 5:
            prevLayer = 5

    wCfg[-1] = 1

    actCfg = []
    for _ in range(layers + 1):
        actCfg.append(random.choice(actlist))

    actCfg[-1] = 'sigmoid' #!!!!!!!!!!!

    opt = random.choice(optlist)

    return (wCfg, actCfg, opt)

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

# X = np.delete(X, 10, 1)
# X = np.delete(X, 9, 1)
# X = np.delete(X, 8, 1)
# X = np.delete(X, 1, 1)

#X = (X - X.min(0)) / X.ptp(0)

# print(X)

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

totalmut = 0.0
posmut = 0.0
totalcross = 0.0
poscross = 0.0

pop = []
for _ in range(popsize):
    cfg = generateConfigForML5()
    fitness = calcFitness(cfg)

    pop.append((fitness, cfg, 0))

hallOfFame = []

for epoch in range(totalEpochs):
    newpop = []
    for p in pop:
        fit, cfg, age = p
        if age < ageOfRemoval:
            good = True
            for newp in newpop:
                tfit, tcfg, tage = newp
                if configDiff(tcfg, cfg) < 3: # diversify population
                    good = False
                    break
            if good:
                newpop.append((fit, cfg, age + 1))

    pop = sorted(newpop)[:popsize / 4]

    hallOfFame.append(pop[0])

    flog.write("epoch: " + str(epoch) + "\n")
    for p in pop:
        flog.write(str(p) + "\n")

    flog.flush()

    print(time.strftime("[ %H:%M:%S ] ") + "cur. best: " + str(pop[0]))

    for _ in range(popsize / 4):
        fit1, cfg1, _ = random.choice(pop)
        fit2, cfg2, _ = random.choice(pop)
        fit3, cfg3, _ = random.choice(pop)

        cfg = crossoverConfigs(cfg1, cfg2, cfg3)

        fit = calcFitness(cfg)

        if fit < min(fit1, fit2, fit3):
            poscross += 1
        totalcross += 1

        pop.append((fit, cfg, 0))

    for _ in range(popsize / 4):
        fitwas, cfg, _ = random.choice(pop)
        cfg = mutateConfig(cfg)
        fit = calcFitness(cfg)

        if fit < fitwas:
            posmut += 1
        totalmut += 1

        pop.append((fit, cfg, 0))

    while len(pop) < popsize:
        cfg = generateConfigForML5()
        fitness = calcFitness(cfg)

        pop.append((fitness, cfg, 0))

    print("mutation: " + str(posmut / totalmut) + " / " + str(totalmut))
    print("cross: " + str(poscross / totalcross) + " / " + str(totalcross))

    k.clear_session()

flog.close()

print("hallOfFame: ")
for f in sorted(hallOfFame):
    print(f)