import theano.tensor as T
import numpy
import theano 
from NNet.Util import defaultGradParameters, WeightTanhGenerator,\
    WeightEqualZeroGenerator, hard_tanh


class HiddenLayer(object):
    def __init__(self, input, numberNeuronsPreviousLayer, numberClasses, W=None, b=None,
                 activation="tanh",weightTanhGenerator= WeightTanhGenerator()):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (numberNeuronsPreviousLayer,numberClasses)
        and the bias vector b is of shape (numberClasses,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, numberNeuronsPreviousLayer)

        :type numberNeuronsPreviousLayer: int
        :param numberNeuronsPreviousLayer: dimensionality of input

        :type numberClasses: int
        :param numberClasses: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        
        :type weightTanhGenerator: WeightGenerator
        :param weightTanhGenerator: The object responsible for generating the values of the weights
        """
        self.input = input
        
        if activation == 'tanh':
            activation = T.tanh
            
        elif activation == 'hard_tanh':
            activation = hard_tanh
            
        elif activation == 'sigmoid':
            activation = T.nnet.sigmoid
        
        elif activation == 'hard_sigmoid':
            activation = T.nnet.hard_sigmoid
        
        elif activation == 'ultra_fast_sigmoid':
            activation = T.nnet.ultra_fast_sigmoid
        
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(numberNeuronsPreviousLayer+n_hidden)) and sqrt(6./(numberNeuronsPreviousLayer+n_hidden))
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
                weightTanhGenerator.generateWeight(numberNeuronsPreviousLayer,numberClasses),
                dtype=theano.config.floatX
            )
            if activation != T.tanh and activation != hard_tanh :
                W_values *= 4

            W = theano.shared(value=W_values, name='W_hiddenLayer', borrow=True)

        if b is None:
            b_values = WeightEqualZeroGenerator().generateWeight(numberClasses)
            b = theano.shared(value=b_values, name='b_hiddenLayer', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
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
    
    def getUpdate(self,cost,learningRate):
        return defaultGradParameters(cost,self.params,learningRate);
    