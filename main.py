from neon.callbacks.callbacks import Callbacks
from neon.data import ArrayIterator
from neon.initializers import Gaussian
from neon.layers import GeneralizedCost, Affine, Linear
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Logistic, Explin, CrossEntropyBinary, Misclassification, Tanh, Softmax, Accuracy
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger
from neon.backends import gen_backend
from neon.layers import Linear, Bias, Recurrent
from neon.transforms import SumSquared, MeanSquared
from neon.transforms import LogLoss, PrecisionRecall, transform

import numpy as np

class TSign(transform.Transform):
    def __init__(self, name=None):
        super(TSign, self).__init__(name)

    # f(x) = max(0,x)
    def __call__(self, x):
        if x > 0:
            return 1
        return -1 #self.be.greater(x, 0)

    # If x > 0, gradient is 1; otherwise 0.
    def bprop(self, x):
        return 0.001

backend = 'gpu'
print("backend: ", backend)

be = gen_backend(backend = backend)

be.bsz = 10000

bitsize = 20
print("Generating dataset...")
#X = np.random.randint(2, size = (3000000, bitsize))
X = np.random.uniform(size = (300000, bitsize))
y = X

train = ArrayIterator(X = X, y = y, make_onehot = False)

print("Init mlp...")

# setup weight initialization function
init_norm = Gaussian(loc=0.0, scale=0.1)

layers = [
            Affine(nout = bitsize , init=init_norm, activation=Tanh()),
            #Affine(nout = bitsize , init=init_norm, activation=Tanh()),
            Affine(nout = 15, init=init_norm, activation=Tanh()),
            Affine(nout = 10, init=init_norm, activation=Tanh()),
            #Affine(nout = bitsize * 2, init=init_norm, activation=Tanh()),
            Affine(nout = 15 , init=init_norm, activation=Rectlin()),
            #Affine(nout = bitsize , init=init_norm, activation=TSign())
            Affine(nout = bitsize , init=init_norm, activation=Logistic())
         ]

# setup cost function
cost = GeneralizedCost(costfunc=SumSquared())

# setup optimizer
optimizer = GradientDescentMomentum(
    0.1, momentum_coef=0.9)

# initialize model object
mlp = Model(layers=layers)

# run fit
mlp.fit(train, 
        callbacks=Callbacks(mlp),
        optimizer=optimizer,
        num_epochs=1200, 
        cost=cost
        )

X = np.random.randint(2, size = (100000, bitsize))
y = X

mlp2 = Model(layers = mlp.layers.layers[:4])


print("len = ", len(mlp.layers.layers))
metric = Accuracy()
print("eval train = ", mlp.eval(train, metric=metric))

train = ArrayIterator(X = X, y = y, make_onehot = False)

#print(mlp.get_outputs(train))
print("eval1 = ", mlp.eval(train, metric=metric))

#print(mlp2.get_outputs(train))
print("eval2 = ", mlp2.eval(train, metric=metric))

mlp3 = Model(layers = mlp.layers.layers + mlp.layers.layers)
print("doubled eval = ", mlp3.eval(train, metric=metric))

mlp4 = Model(layers = mlp2.layers.layers + mlp3.layers.layers)
print("eval4= ", mlp4.eval(train, metric=metric))