import numpy
import theano

from NNet.Layer import Layer
import theano.tensor as T


class SoftmaxLayer(Layer):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, _input, lenIn, lenOut):
        """ Initialize the parameters of the logistic regression

        :type _input: theano.tensor.TensorType
        :param _input: symbolic variable that describes the _input of the
                      architecture (one minibatch)

        :type lenIn: int
        :param lenIn: number of _input units, the dimension of the space in
                     which the datapoints lie

        :type lenOut: int
        :param lenOut: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (lenIn, lenOut)
        Layer.__init__(self, _input)
        
        self.W = theano.shared(
            value=numpy.zeros(
                (lenIn, lenOut),
                dtype=theano.config.floatX
            ),
            name='W_softmax',
            borrow=True
        )
        # initialize the baises b as a vector of lenOut 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (lenOut,),
                dtype=theano.config.floatX
            ),
            name='b_softmax',
            borrow=True
        )

        #
        # Symbolic expression for computing the matrix of
        #     class-membership probabilities, where:
        #
        #     W is a matrix where column-k represent the separation hyper plan for class-k,
        #     x is a matrix where row j represents training sample j, and
        #     b is a vector where element k represent the free parameter of hyper plan k.
        #
        self.p_y_given_x = T.nnet.softmax(T.dot(_input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
    
    def getOutput(self):
        return self.p_y_given_x
    
    def getParameters(self):
        return self.params
    
    def getPrediction(self):
        return self.y_pred
    
    def getDefaultGradParameters(self):
        return self.params
    
    def getUpdates(self, cost, learningRate):
        return []
