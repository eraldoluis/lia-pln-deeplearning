import theano.tensor as T
import numpy
import theano 
from NNet.Util import WeightTanhGenerator, WeightEqualZeroGenerator, hard_tanh
from NNet.Layer import Layer

class HiddenLayer(Layer):
    def __init__(self, _input, lenIn, lenOut, W=None, b=None,
                 activation="tanh", weightTanhGenerator=WeightTanhGenerator()):
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
        
        :type weightTanhGenerator: WeightGenerator
        :param weightTanhGenerator: The object responsible for generating the values of the weights
        """
        Layer.__init__(self, _input)
        
        if activation == 'tanh':
            activation = T.tanh
            
        elif activation == 'hard_tanh':
            activation = hard_tanh
            
        elif activation == 'sigmoid':
            activation = T.nnet.sigmoid
        
        elif activation == 'hard_sigmoid':
            activation = T.nnet.hard_sigmoid
        
        
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(lenIn+n_hidden)) and sqrt(6./(lenIn+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                weightTanhGenerator.generateWeight(lenIn, lenOut),
                dtype=theano.config.floatX
            )
            if activation != T.tanh and activation != hard_tanh :
                W_values *= 4

            W = theano.shared(value=W_values, name='W_hiddenLayer', borrow=True)

        if b is None:
            b_values = WeightEqualZeroGenerator().generateWeight(lenOut)
            b = theano.shared(value=b_values, name='b_hiddenLayer', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(_input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
    
    def getOutput(self):
        return self.output 
    
    def getParameters(self):
        return self.params
    
    def getDefaultGradParameters(self):
        return self.params
    
    def getStructuredParameters(self):
        return []

    def getUpdates(self, cost, lr, sumSqGrads=None):
        return []
