import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution1D, GlobalMaxPooling1D, MaxPooling1D, AveragePooling1D, Reshape, Flatten
from keras.models import load_model
from keras import optimizers

import copy
import random

ltypes = [ 'Dense', 'Convolution1D', 'MaxPooling1D', 'Dropout' ]

ilayers = [ 'Dense', 'Convolution1D' ]

olayers = [ 'Dense' ]

actlist = [ 'tanh', 'sigmoid', 'relu', 'softplus' ]

optlist = [ 'sgd', 'rmsprop', 'adam', 'adamax', 'adadelta', 'adagrad' ]

class ModelConfig:
    def __str__(self):
        return "( " + str(self.layerType) + ", " + str(self.layerParam) + ", " + str(self.layerIsActive) + ", " + str(self.opt) + ")"

    def __init__(self, arg = None):
        self.layerType = []
        self.layerParam = []
        self.layerIsActive = []
        self.inputSize = 11
        self.outputSize = 1
        self.opt = None
        
        if arg != None:
            self.layerType, self.layerParam, self.layerIsActive = copy.deepcopy(arg)

    def addRandomLayer(self, nout = None, maxnout = None, alist = None, llist = None):
        if maxnout == None:
            maxnout = 100

        if nout == None:
            nout = random.randint(1, maxnout)

        if alist == None:
            alist = actlist

        if llist == None:
            llist = ltypes

        ltype = random.choice(llist)
        self.layerType.append(ltype)

        if ltype == 'Dense':
            self.layerParam.append( (nout, random.choice(alist) ) )

        if ltype == 'Dropout':
            self.layerParam.append( random.uniform(0.0, 1.0) )

        if ltype == 'Convolution1D':
            self.layerParam.append( (nout, random.randint(1, 10), random.choice(alist) ) )

        if ltype == 'MaxPooling1D':
            self.layerParam.append( (nout, random.randint(1, 10)) )

        self.layerIsActive.append(random.choice([True, False]))

    def createKerasModel(self):
        print self.__str__()

        model = Sequential()

        ltype = self.layerType[0]
        param = self.layerParam[0]

        prevShape = 0

        if ltype == 'Dense':
            nout, act = param
            model.add(Dense(nout, activation = act, input_dim = self.inputSize))
            prevShape = 2
            prevOut = nout

        if ltype == 'Convolution1D':
            model.add(Reshape((self.inputSize, 1), input_shape = (self.inputSize, )))
            nout, kernel_sz, act = param
            model.add(Convolution1D(nout, activation = act, kernel_size = kernel_sz, strides = 1, input_shape = (None, self.inputSize)))
            prevShape = 3
            prevOut = nout

        if ltype == 'MaxPooling1D':
            model.add(Reshape((self.inputSize, 1), input_shape = (self.inputSize, )))
            nout, poolSize = param
            model.add(MaxPooling1D(pool_size = poolSize, input_dim = self.inputSize))
            prevShape = 3
            prevOut = nout

        for ltype, param, active in zip(self.layerType[1:-1], self.layerParam[1:-1], self.layerIsActive[1:-1]):
            if active:
                if ltype == 'Dense':
                    if prevShape != 2:
                        model.add(Flatten())
                        prevShape = 2
                    nout, act = param
                    model.add(Dense(nout, activation = act))
                    prevOut = nout

                if ltype == 'Convolution1D':
                    if prevShape != 3:
                        model.add(Reshape((prevOut, 1)))
                        prevShape = 3
                    nout, kernel_sz, act = param
                    model.add(Convolution1D(nout, activation = act, kernel_size = min(kernel_sz, prevOut), strides = 1))
                    prevOut = nout

                if ltype == 'MaxPooling1D':
                    if prevShape != 3:
                        model.add(Reshape((prevOut, 1)))
                        prevShape = 3
                    nout, poolSize = param
                    model.add(MaxPooling1D(pool_size = min(poolSize, prevOut)))
                    prevOut = nout

                if ltype == 'Dropout':
                    model.add(Dropout(param))
        if prevShape != 2:
            model.add(Flatten())
        nout, act = self.layerParam[-1]
        model.add(Dense(self.outputSize, activation = act))

        model.compile(
            loss='binary_crossentropy',
            optimizer=self.opt
        )

        return model


def generateModelConfig(layers = 6, inputSize = 11, outputSize = 1):
    cfg = ModelConfig()
    print cfg
    cfg.inputSize = inputSize
    cfg.outputSize = outputSize

    cfg.opt = random.choice(optlist)

    cfg.addRandomLayer(llist = ilayers)

    for _ in range(layers - 2):
        cfg.addRandomLayer()

    cfg.addRandomLayer(llist = olayers, nout = outputSize)

    cfg.layerIsActive[0] = True
    cfg.layerIsActive[-1] = True    

    return cfg

