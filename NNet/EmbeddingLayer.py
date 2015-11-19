import theano.tensor as T
from NNet.Layer import Layer

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

    def __init__(self, examples, embedding):
        """
        :type examples: T.TensorType
        :param examples: matrix of examples (mini-batch).
            Each examples is an array of feature values.
            Each feature value is an index within the given embedding.
        
        :type embedding: T.TensorSharedVariable
        :param embedding: This is the dictionary of feature embeddings that
            comprise this layer parameters.
            It is a matrix of parameters.
            Each row corresponds to a feature value weight vector
                (this feature embedding).
        
        In the following, we describe the shape of these two variables.
        
        numExs: number of examples (examples.shape[0])
        
        szEx: size of each example (examples.shape[1]), or number of features.

        numVectors: number of vectors in the embedding (embedding.shape[0]), or 
            size of the vocabulary of features (number of possible feature 
            values).

        szEmb: size of the embedding (embedding.shape[1]), number of parameters
            to represent each feature value.

        examples.shape = (numExs, szEx)
        
        embedding.shape = (numVectors, szEmb)

        """
        Layer.__init__(self, examples)
        
        self.__embedding = embedding

        # Matrix of the active parameters for the given examples.
        # Its shape is (numExs * szEx, szEmb).
        self.__activeVectors = embedding[examples.flatten(1)]

        #
        # Output of the layer for the given examples.
        # Its shape is (numExs, szEx * szEmb).
        #
        # This variable holds the same information as self.__activeVectors,
        # but with a different shape.
        #
        self.__output = embedding[examples].flatten(2)

    def getOutput(self):
        return self.__output
    
    def getParameters(self):
        return self.__embedding

    def getUpdates(self, cost, learningRate):
        # shape = (numExs, szEx * szEmb)
        gWordVector = -learningRate * T.grad(cost, self.__output)

        # Reshape gradient vector as self.__activeVectors, since these are the
        # parameters to be updated.
        gWordVector = gWordVector.reshape(self.__activeVectors.shape)

        # Update only the active vectors.
        up = T.inc_subtensor(self.__activeVectors, gWordVector)

        return [(self.__embedding, up)]

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
