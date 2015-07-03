#import theano
import theano.tensor as T
import numpy


################################ FUNCTIONS ########################
def defaultGradParameters(cost, parameters, learningRate):
    gparams = [T.grad(cost, param) for param in parameters]

    return [
               (param, param - learningRate * gparam)
               for param, gparam in zip(parameters, gparams)
              ]     


def negative_log_likelihood(output, y):
    """Return the mean of the negative log-likelihood of the prediction
    of this model under a given target distribution.

    .. math::

        \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
        \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
            \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
        \ell (\theta=\{W,b\}, \mathcal{D})
    
    
    :type output: theano.tensor.Variable
    :param output: output of layer
              
    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
              correct label

    Note: we use the mean instead of the sum so that
          the learning rate is less dependent on the batch size
    """
    return -T.mean(T.log(output)[T.arange(y.shape[0]), y])
    
def regularizationSquareSumParamaters(parameters,regularizationFactor,numberExamples):
    p = 0
    
    for par in parameters:
        p += T.sum(T.pow(par,2));
    
    return regularizationFactor * p / (2 * numberExamples)

################################ OBJECTS ########################

class WeightTanhGenerator:
    
    def generateWeight(self,fin, fout):
        eInit = self.getEInit(fin ,fout)
        
        return  numpy.random.random_sample((fin, fout)) * 2 * eInit - eInit
    
    def generateVector(self,num):
        eInit = self.getEInit(num, 0);
        
        return  numpy.random.random_sample(num) * 2 * eInit - eInit
    
    def getEInit(self,fin,fout):
        return numpy.sqrt(6. / (fin + fout))

class LearningRateUpdNormalStrategy:
    
    def getCurrentLearninRate(self,learningRate,numEpoch):
        return learningRate

class LearningRateUpdDivideByEpochStrategy:
    
    def getCurrentLearninRate(self,learningRate ,numEpoch):
        return learningRate/numEpoch

    