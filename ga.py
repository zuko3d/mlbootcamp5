from neon.callbacks.callbacks import Callbacks
from neon.data import ArrayIterator
from neon.initializers import Gaussian, Uniform
from neon.layers import GeneralizedCost, Affine, Linear, GRU, LSTM
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp
from neon.transforms import Rectlin, Logistic, Explin, CrossEntropyBinary, Misclassification, Tanh, Softmax, Accuracy
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger
from neon.backends import gen_backend
from neon.layers import Linear, Bias, Recurrent
from neon.transforms import SumSquared, MeanSquared
from neon.transforms import LogLoss, PrecisionRecall, transform

import numpy as np

import csv

backend = 'gpu'
print("backend: ", backend)

be = gen_backend(backend = backend)

be.bsz = 5000

print("Reading dataset...")

csv.register_dialect('semicolon', delimiter=';')

fin = open("train.csv", 'rt')
reader = csv.reader(fin, dialect='semicolon')

data = [d for d in reader]
data = data[1:5001]
#data = data[1:]

fin.close()

X = np.asarray(data, dtype = np.float64)

X = np.delete(X, 0, 1)

Y = X[:, [11]]
X = np.delete(X, 11, 1)

# X = np.delete(X, 10, 1)
# X = np.delete(X, 9, 1)
# X = np.delete(X, 8, 1)
# X = np.delete(X, 1, 1)

X = (X - X.min(0)) / X.ptp(0)

print(X)

#Y = Y * 2 - 1

truth = np.copy(Y)

z = np.copy(Y)
z = 1 - z

Y = np.concatenate((Y,z), axis = 1)

print(Y)

train = ArrayIterator(X = X, y = Y, make_onehot = False)

print("Init mlp...")

# setup weight initialization function
#init_norm = Gaussian(loc=0.0, scale=1)
init_norm = Uniform(low=-2, high=2)

layers = [
            Affine(nout = 50 , init=init_norm, activation=Rectlin()),
            Affine(nout = 20 , init=init_norm, activation=Tanh()),
            #Affine(nout = 400 , init=init_norm, activation=Rectlin()),
            Affine(nout = 4 , init=init_norm, activation=Softmax()),
            Affine(nout = 3 , init=init_norm, activation=Logistic()),
            Affine(nout = 2 , init=init_norm, activation=Softmax())
         ]

# setup cost function
cost = GeneralizedCost(costfunc=CrossEntropyBinary())
#cost = GeneralizedCost(costfunc=MeanSquared())
# CrossEntropyBinary

# setup optimizer
optimizer = GradientDescentMomentum(
        learning_rate = 0.01,
        momentum_coef = 0.9
        )

# initialize model object
mlp = Model(layers=layers)

callbacks = Callbacks(mlp)
callbacks.add_save_best_state_callback("early_stop-best_state.pkl")

# run fit
mlp.fit(train, 
        callbacks=callbacks,
        optimizer=optimizer,
        num_epochs=2500, 
        cost=cost
        )

metric = Misclassification()
print("Misclassification on train = ", mlp.eval(train, metric=metric))