# Activation Layer
import numpy
import theano
import theano.tensor as T
from theano.tensor.sharedvar import TensorSharedVariable

from nnet.Layer import Layer

""
class DotLayer(Layer):
    def __init__(self, layer1, layer2, transposeLayer2=True, b=None, useDiagonal=False):
        inputs = [layer1, layer2]

        if isinstance(b, (Layer)):
            inputs.append(b)

        super(DotLayer, self).__init__(inputs)

        if transposeLayer2:
            output2 = layer2.getOutput().T
        else:
            output2 = layer2.getOutput()

        output = T.dot(layer1.getOutput(), output2)


        if useDiagonal:
            output = T.diagonal(output)

        if b:
            if isinstance(b, (Layer)):
                b = b.getOutput()
            elif not isinstance(b, TensorSharedVariable):
                if isinstance(b, (numpy.ndarray, list)):
                    b_values = numpy.asarray(b, dtype=theano.config.floatX)
                else:
                    # b_values = numpy.zeros(bSize, dtype=theano.config.floatX)
                    raise Exception("b is not a list or theano variable")
                b = theano.shared(value=b_values, name='b_hiddenLayer', borrow=True)

            output += b

        self.__output = output


    def getOutput(self):
        return self.__output

    def getParameters(self):
        return []

    def getStructuredParameters(self):
        return []

    def getDefaultGradParameters(self):
        return []

    def getUpdates(self, cost, learningRate, sumSqGrads=None):
        return []
