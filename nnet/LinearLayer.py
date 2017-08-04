import numpy
import theano
import theano.tensor as T
from theano.tensor.sharedvar import TensorSharedVariable

from nnet.Layer import Layer
from nnet.WeightGenerator import GlorotUniform


class LinearLayer(Layer):
    def __init__(self, _input, lenIn, lenOut, W=None, b=None, weightInitialization=GlorotUniform(), trainable=True,
                 name=None):
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

        :param trainable: set if the layer is trainable or not

        :param name: unique name of the layer. This is use to save the attributes of this object.
        """
        super(LinearLayer, self).__init__(_input, trainable, name)

        if not isinstance(W, TensorSharedVariable):
            if W is None:
                # Initialize according to the given initialization method.
                W = numpy.asarray(weightInitialization.generateWeight((lenIn, lenOut)), dtype=theano.config.floatX)
            elif isinstance(W, (numpy.ndarray, list)):
                # Initialize with the given values (numpy array or list of values).
                W = numpy.asarray(W, dtype=theano.config.floatX)
            else:
                # Unknown value provided.
                raise Exception("Provided 'W' value is not from a supported type!")

            W = theano.shared(value=W, name='W_hiddenLayer', borrow=True)

        if not isinstance(b, TensorSharedVariable):
            if b is None:
                b = numpy.zeros(lenOut, dtype=theano.config.floatX)
            elif isinstance(b, (numpy.ndarray, list)):
                b = numpy.asarray(b, dtype=theano.config.floatX)
            else:
                raise Exception("Provided 'b' value is not from a supported type!")

            b = theano.shared(value=b, name='b_hiddenLayer', borrow=True)

        self.W = W
        self.b = b

        # Output variable.
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

    @staticmethod
    def getParametersFromPersistenceManager(persistenceManager, name):
        """
        Return parameters of a linear alayer from the database

        :type persistenceManager: persistence.PersistentManager.PersistentManager
        :param persistenceManager:

        :param name: name of object which parameters was saved as attribute

        :return:  a tuple with  W e b values
        """
        attrs = persistenceManager.getObjAttributesByObjName(name)
        return numpy.array(attrs["w"]), numpy.array(attrs["b"])

    def getAttributes(self):
        return {
            "w": self.W.get_value(),
            "b": self.b.get_value()
        }

    def load(self, attributes):
        self.W.set_value(numpy.array(attributes["w"]))
        self.b.set_value(numpy.array(attributes["b"]))
