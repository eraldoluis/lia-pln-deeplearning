import theano.tensor as T
import numpy
import theano
from keras.layers.core import Dense

from NNet.Util import WeightTanhGenerator, WeightEqualZeroGenerator, hard_tanh
from NNet.Layer import Layer
from NNet.WeightGenerator import GlorotUniform


class LinearLayer(Layer):
    def __init__(self, lenIn, lenOut, W=None, b=None, weightInitialization=GlorotUniform()):
        """
        Typical hidden layer of a MLP: units are fully-connected.
        Weight matrix W is of shape (lenIn,lenOut) 
            and the bias vector b is of shape (lenOut,).

        The nonlinearity used here is tanh, for default. But it can be given
        any alternative function.

        Hidden unit activation is given by: fact(dot(_input,W) + b)

        :type _input: theano.tensor.dmatrix
        :param _input: a symbolic tensor of shape (n_examples, lenIn)

        :type lenIn: int
        :param lenIn: dimensionality of _input

        :type lenOut: int
        :param lenOut: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        
        :type weightInitialization: NNet.WeightGenerator.WeightGenerator
        """
        Layer.__init__(self)

        if W is None:
            W_values = numpy.asarray(
                weightInitialization.generateWeight((lenIn, lenOut)),
                dtype=theano.config.floatX
            )

            W = theano.shared(value=W_values, name='W_hiddenLayer', borrow=True)

        if b is None:
            b_values = WeightEqualZeroGenerator().generateWeight(lenOut)
            b = theano.shared(value=b_values, name='b_hiddenLayer', borrow=True)

        self.W = W
        self.b = b

        # parameters of the model
        self.params = [self.W, self.b]


    def getOutput(self,x):
        return T.dot(x, self.W) + self.b

    def getParameters(self):
        return self.params

    def getDefaultGradParameters(self):
        return self.params

    def getStructuredParameters(self):
        return []

    def getUpdates(self, cost, lr, sumSqGrads=None):
        return []
