import theano.tensor as T
import numpy
import theano
from NNet.Util import defaultGradParameters

class SoftmaxLayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, numberNeuronsPreviousLayer, numberClasses):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type numberNeuronsPreviousLayer: int
        :param numberNeuronsPreviousLayer: number of input units, the dimension of the space in
                     which the datapoints lie

        :type numberClasses: int
        :param numberClasses: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (numberNeuronsPreviousLayer, numberClasses)
        
        
        self.W = theano.shared(
            value=numpy.zeros(
                (numberNeuronsPreviousLayer, numberClasses),
                dtype=theano.config.floatX
            ),
            name='W_softmax',
            borrow=True
        )
        # initialize the baises b as a vector of numberClasses 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (numberClasses,),
                dtype=theano.config.floatX
            ),
            name='b_softmax',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]
    
    def getOutput(self):
        return self.p_y_given_x;
    
    def getParameters(self):
        return self.params
    
    def getPrediction(self):
        return self.y_pred;
    
    def getUpdate(self,cost,learningRate):
        return defaultGradParameters(cost,self.params,learningRate);

