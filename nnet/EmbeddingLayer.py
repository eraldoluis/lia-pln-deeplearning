import numpy

import theano
import theano.tensor as T
from theano.tensor.sharedvar import TensorSharedVariable

from nnet.Layer import Layer


class EmbeddingLayer(Layer):
    """
    Layer that represents a lookup table for some feature embedding.
    It converts a set of sparse features (binary, categorical, etc.)
        to dense vectors.
    Its input is a list of examples (a batch or mini-batch),
        where each example is a list of feature values
            that are active within the example.
    Each feature value corresponds to the index within a feature embedding.
    The embedding is a dictionary of feature (dense) vectors.
    These vectors comprise the learnable parameters of this layer.
    """

    def __init__(self, _input, embedding, borrow=True, structGrad=True, trainable=True, name=None):
        """
        :param _input: _input need be a matrix with 2 dimensions

        :type embedding: numpy.array or python list
        :param embedding: This is the dictionary of feature embeddings, which
            comprise this layer parameters.
            It is a matrix of parameters.
            Each row corresponds to a weight vector that represents a feature
            value (this feature embedding). For instance, when using word
            embeddings, each row stores a word vector.

        :param borrow: borrow attribute when creating the shared variable for the given embedding matrix.
        
        :param structGrad: whether to use structured gradients or not.
            When using small batches (online gradient descent, in the limit),
            the structured gradient is much more efficient because a small
            fraction of word vectors are used on each iteration.
            However, when using large batches (ordinary gradient descent, in the
            limit), ordinary gradients and updates are more efficient because
            most (or all of the) word vectors are used on each iteration.

        :param trainable: set if the layer is trainable or not

        :param name: unique name of the layer. This is use to save the attributes of this object.

        Considering that:
        
            numExs: number of examples (examples.shape[0])
            
            szEx: size of each example (examples.shape[1]), or number of 
                features.
    
            numVectors: number of vectors in the embedding (embedding.shape[0]), 
                or size of the vocabulary of features (number of possible 
                feature values).
    
            szEmb: size of the embedding (embedding.shape[1]), number of 
                parameters to represent each feature value.

        We can define the shape of the two first parameters as:

            examples.shape = (numExs, szEx)
        
            embedding.shape = (numVectors, szEmb)

        """
        super(EmbeddingLayer, self).__init__(_input, trainable, name)

        if not isinstance(embedding, TensorSharedVariable):
            if isinstance(embedding, list):
                embedding = numpy.asarray(embedding, dtype=theano.config.floatX)

            self.__embedding = theano.shared(value=embedding, name='embedding', borrow=borrow)
        else:
            self.__embedding = embedding

        # Whether to use structured gradients or not.
        self.__structGrad = structGrad

        # Matrix of the active parameters for the given examples.
        # Its shape is (numExs * szEx, szEmb).
        flat = T.flatten(_input, 1)
        self.__activeVectors = self.__embedding[flat]

        #
        # Output of the layer for the given examples.
        # Its shape is (numExs, szEx, szEmb).
        #
        # This variable holds the same information as self.__activeVectors,
        # but with a different shape.
        #
        self.__output = self.__activeVectors.reshape((_input.shape[0], _input.shape[1], -1))

    def getUpdates(self, cost, learningRate, sumSqGrads=None):
        if self.__structGrad:
            # shape = (numExs, szEx * szEmb)
            grad = T.grad(cost, self.__activeVectors)

            # Reshape the gradient vector as self.__activeVectors, since these 
            # are the parameters to be updated, which follow the shape of the
            # embedding itself (in fact, a subtensor of the embedding).
            # grad = grad.reshape(self.__activeVectors.shape)

            # List of updates.
            updates = []

            # Whether to use AdaGrad or not.
            if sumSqGrads:
                # Each layer gives a list of variables (one for each parameter).
                # Since this layer has only one parameter variable (the embedding),
                # we access only the first (and only) element of the list.
                sumSqGrads = sumSqGrads[0]
                # For numerical stability.
                fudgeFactor = 1e-10
                # Select only rows activated by the given input.
                sumSqGradsSub = sumSqGrads[self.getInput().flatten(1)]
                # Update of the sum of squared historical gradients.
                gradSq = grad * grad
                newSsg = T.inc_subtensor(sumSqGradsSub, gradSq)
                updates.append((sumSqGrads, newSsg))
                # Update of the parameter.
                newParam = T.inc_subtensor(self.__activeVectors,
                                           - learningRate * (grad / (fudgeFactor + T.sqrt(sumSqGradsSub + gradSq))))
                updates.append((self.__embedding, newParam))
            else:
                # Update only the active vectors (structured update).
                up = T.inc_subtensor(self.__activeVectors, -learningRate * grad)
                updates = [(self.__embedding, up)]

            return updates

        # When using ordinary gradients, we need to return these parameters
        # through the getDefaultGradParameters(...) method.
        return []

    def getNormalizationUpdate(self, strategy, normFactor):
        """
        :return a list of updates that normalize the parameters according to 
            the given strategy.
        """
        emb = self.__embedding

        if strategy == "minmax":
            norm = normFactor * (emb - emb.min(axis=0)) / emb.ptp(axis=0)
        elif strategy == "zscore":
            norm = normFactor * (emb - emb.mean(axis=0)) / emb.std(axis=0)
        elif strategy == "sphere":
            norm = normFactor * (emb / (emb ** 2).sum(axis=0).sqrt())
        else:
            raise Exception("Unknown normalization strategy: %s" % strategy)

        return [(emb, norm)]

    def getOutput(self):
        return self.__output

    def getParameters(self):
        return [self.__embedding]

    def getStructuredParameters(self):
        if self.__structGrad:
            return [self.__embedding]
        return []

    def getDefaultGradParameters(self):
        """
        Since this layer uses a structured update for all its parameters
        (embedding), there is no parameter with default gradient updates.
        Unless the flag self.__structGrad is equal to False.
        """
        if self.__structGrad:
            return []

        return [self.__embedding]

    def getAttributes(self):
        return {
            "emb": self.__embedding.get_value()
        }

    def load(self, attributes):
        self.__embedding.set_value(numpy.array(attributes["emb"]))

    @staticmethod
    def getEmbeddingFromPersistenceManager(persistenceManager, name):
        """
        Return the embedding vector from the database

        :type persistenceManager: persistence.PersistentManager.PersistentManager
        :param persistenceManager:

        :param name: name of object which the embedding was saved as attribute

        :return:
        """
        attrs = persistenceManager.getObjAttributesByObjName(name)

        return attrs["emb"]
