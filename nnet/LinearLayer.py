import theano.tensor as T
import numpy
import theano
from theano.tensor.sharedvar import TensorSharedVariable

from nnet.Layer import Layer
from nnet.WeightGenerator import GlorotUniform


class LinearLayer(Layer):
    def __init__(self, _input, lenIn, lenOut, W=None, b=None, weightInitialization=GlorotUniform()):
        """
        Typical linear layer of a MLP: units are fully-connected.
        Weight matrix W is of shape (lenIn,lenOut) 
            and the bias vector b is of shape (lenOut,).

        Hidden unit activation is given by: dot(_input,W) + b

        :param _input: a layer or theano variable

        :type lenIn: int
        :param lenIn: dimensionality of _input

        :type lenOut: int
        :param lenOut: number of hidden units

        :type weightInitialization: nnet.WeightGenerator.WeightGenerator
        """
        super(LinearLayer, self).__init__(_input)

        if not isinstance(W, TensorSharedVariable):
            if isinstance(W, (numpy.ndarray, list)):
                W_values = numpy.asarray(W, dtype=theano.config.floatX)
            else:
                W_values = numpy.asarray(
                    weightInitialization.generateWeight((lenIn, lenOut)),
                    dtype=theano.config.floatX
                )

            W = theano.shared(value=W_values, name='W_hiddenLayer', borrow=True)

        if not isinstance(b,TensorSharedVariable):
            if isinstance(b, (numpy.ndarray, list)):
                b_values = numpy.asarray(b, dtype=theano.config.floatX)
            else:
                b_values = numpy.zeros(lenOut, dtype=theano.config.floatX)

            b = theano.shared(value=b_values, name='b_hiddenLayer', borrow=True)

        self.W = W
        self.b = b

        self.__output = T.dot(self.getInput(), self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]

    def getOutput(self):
        return self.__output

    def getParameters(self):
        return self.params

    def getDefaultGradParameters(self):
        return self.params

    def getStructuredParameters(self):
        return []

    def getUpdates(self, cost, lr, sumSqGrads=None):
        return []
