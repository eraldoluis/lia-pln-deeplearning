import theano
import theano.tensor as T
import numpy

################################ FUNCTIONS ########################





def defaultGradParameters(cost, parameters, learningRate, sumsSqGrads=None):
    """
    :param cost: symbolic variable expressing the cost function value.

    :param parameters: shared variable with the parameters to compute the gradient w.r.t.

    :param learningRate: value or symbolic variable representing the learning rate.

    :param sumsSqGrads: (optional) shared variable storing the sum of the
        squared historical gradient for each parameter,
        which are used in AdaGrad.
    """
    # Compute gradient of the cost function w.r.t. each parameter.
    grads = [T.grad(cost, param) for param in parameters]

    if sumsSqGrads:
        # For numerical stability.
        fudgeFactor = 1e-10
        updates = []
        for param, grad, ssg in zip(parameters, grads, sumsSqGrads):
            # Update of the sum of squared gradient.
            newSsg = ssg + grad * grad
            updates.append((ssg, newSsg))
            # Update of the parameter.
            newParam = param - learningRate * (grad / (fudgeFactor + T.sqrt(newSsg)))
            updates.append((param, newParam))
    else:
        # Ordinary SGD updates.
        updates = [(param, param - learningRate * grad)
                   for param, grad in zip(parameters, grads)]

    return updates


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
    
def regularizationSquareSumParamaters(parameters, regularizationFactor, numberExamples):
    p = 0
    
    for par in parameters:
        p += T.sum(T.pow(par, 2));
    
    return regularizationFactor * p / (2 * numberExamples)



def hard_tanh(x):
    x = T.switch(x < -1.0, -1.0, x)
    return T.switch(x > 1.0, 1.0, x)


################################ OBJECTS ########################


def generateRandomNumberUniformly(low, high, n_in, n_out):
    if n_out == 0.0:
        return numpy.random.uniform(low, high, (n_in))
    else:
        return numpy.random.uniform(low, high, (n_in, n_out))

class WeightTanhGenerator:
    def generateWeight(self, n_in, n_out):
        high = numpy.sqrt(6. / (n_in + n_out))
        return generateRandomNumberUniformly(-high, high, n_in, n_out)
    
class WeightBottou88Generator:
    
    def generateWeight(self, n_in, n_out=0.0):
        high = 2.38 / numpy.sqrt(n_in)
        
        return generateRandomNumberUniformly(-high, high, n_in, n_out)
    


class WeightEqualOneGenerator:
    
    def generateWeight(self, n_in, n_out=0.0):  
        if n_out == 0.0:
            return numpy.ones(n_in, dtype=theano.config.floatX)
        else:
            return numpy.ones((n_in, n_out), dtype=theano.config.floatX)
    
class FeatureVectorsGenerator:
    
    def generateVector(self, num_features, min_value=-0.1, max_value=0.1):
        return  (max_value * 2) * numpy.random.random_sample(num_features) + min_value

class LearningRateUpdNormalStrategy:
    
    def getCurrentLearninRate(self, learningRate, numEpoch):
        return learningRate

class LearningRateUpdDivideByEpochStrategy:
    
    def getCurrentLearninRate(self, learningRate , numEpoch):
        return learningRate / numEpoch
