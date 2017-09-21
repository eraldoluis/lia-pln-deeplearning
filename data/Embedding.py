#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes and functions that operate on distributed representations
"""

import codecs
import logging
import numpy as np

import theano

from data.Lexicon import Lexicon
from nnet.Util import FeatureVectorsGenerator


#####################################################################
# Embedding classes
####################################################################

class Embedding(object):
    """
    Represents an object distributed representation.
    This class has a matrix with all vectors and lexicon.
    """

    def __init__(self, lexicon, vectors=None, embeddingSize=None):
        """
        Creates a embedding object from lexicon and vectors.
        If vectors is none, so each word in the lexicon will be represented by a random vector with embeddingSize dimensions.
        :type lexicon: data.Lexicon.Lexicon
        :params lexicon: a Lexicon object
        :type vectors: [[int]] | numpy.array | None
        :params vectors: embedding list
        :params embeddingSize: the number of dimensions of vectors. This only will be used when the vectors is none
        """
        self.__lexicon = lexicon

        if not vectors:
            generatorWeight = FeatureVectorsGenerator()
            numVectors = lexicon.getLen()
            vectors = []

            for _ in xrange(numVectors):
                vec = generatorWeight.generateVector(embeddingSize)
                vectors.append(vec)

        self.__vectors = np.asarray(vectors, dtype=theano.config.floatX)
        self.__embeddingSize = self.__vectors.shape[1]

        if lexicon.getLen() != self.__vectors.shape[0]:
            raise Exception("The number of embeddings is different of lexicon size ")

        lexicon.stopAdd()

        if not lexicon.isReadOnly():
            raise Exception(
                "It's possible to insert in the lexicon. Please, transform the lexicon to only read.")

    def exist(self, obj):
        return self.__lexicon.exist(obj)

    def getLexiconIndex(self, obj):
        return self.__lexicon.getLexiconIndex(obj)

    def getEmbeddingByIndex(self, idx):
        return self.__vectors[idx]

    def getEmbedding(self, obj):
        idx = self.__lexicon.getLexiconIndex(obj)
        return self.getEmbeddingByIndex(idx)

    def getEmbeddingMatrix(self):
        return self.__vectors

    def getNumberOfVectors(self):
        return len(self.__vectors)

    def getEmbeddingSize(self):
        return self.__embeddingSize

    def getLexicon(self):
        """
        :return data_operation.lexicon.Lexicon
        """
        return self.__lexicon

    def zscoreNormalization(self, norm_coef=1.0):
        """
        Normalize all the embeddings using the following equation:
        x = (x − mean(x)) / stddev(x)
        :return: None
        """
        self.__vectors = np.asarray(self.__vectors, dtype=theano.config.floatX)

        self.__vectors -= np.mean(self.__vectors, axis=0)
        self.__vectors *= (norm_coef / np.std(self.__vectors, axis=0))

    def minMaxNormalization(self, norm_coef=1.0):
        """
        Normalize all the embeddings to a range [0,1].
        x = (x − min(x)) / (max(x) − min(x))
        :return:None
        """
        self.__vectors = np.asarray(self.__vectors, dtype=theano.config.floatX)

        self.__vectors -= np.min(self.__vectors, axis=0)
        self.__vectors *= (norm_coef / np.ptp(self.__vectors, axis=0))

    def meanNormalization(self, norm_coef=1.0):
        """
        Normalize all the embeddings to a range [-1,1].
        x = (x − mean(x)) / (max(x) − min(x))
        :return:None
        """
        self.__vectors = np.asarray(self.__vectors, dtype=theano.config.floatX)

        self.__vectors -= np.mean(self.__vectors, axis=0)
        self.__vectors *= (norm_coef / np.ptp(self.__vectors, axis=0))

    @staticmethod
    def fromWord2Vec(w2vFile, unknownSymbol, lexiconName=None):
        """
        Creates  a lexicon and a embedding from word2vec file.
        :param w2vFile: path of file
        :param unknownSymbol: the string that represents the unknown words.
        :return: (data.Lexicon.Lexicon, Embedding)
        """
        log = logging.getLogger(__name__)
        fVec = codecs.open(w2vFile, 'r', 'utf-8')

        # Read the number of words in the dictionary and the embedding size
        nmWords, embeddingSizeStr = fVec.readline().strip().split(" ")
        embeddingSize = int(embeddingSizeStr)
        lexicon = Lexicon(unknownSymbol, lexiconName)

        # The empty array represents the array of unknown
        # At end, this array will be replaced by one array that exist in the  w2vFile or a random array.
        vectors = [[]]
        nmEmptyWords = 0

        unknownInsered = False

        for line in fVec:
            splitLine = line.rstrip().split(u' ')
            word = splitLine[0]

            if len(word) == 0:
                log.warning("Insert in the embedding a empty string. This embeddings will be thrown out.")
                nmEmptyWords += 1
                continue

            vec = [float(num) for num in splitLine[1:]]


            if word == unknownSymbol:
                if len(vectors[0]) != 0:
                    raise Exception("A unknown symbol was already inserted.")

                vectors[0] = vec
                unknownInsered = True
            else:
                lexicon.put(word)
                vectors.append(vec)

        if len(vectors[0]) == 0:
            vectors[0] = FeatureVectorsGenerator().generateVector(embeddingSize)

        expected_size = lexicon.getLen() - 1 + nmEmptyWords

        if(unknownInsered):
            expected_size += 1

        if int(nmWords) != expected_size:
            print int(nmWords)
            print expected_size
            print unknownInsered
            raise Exception("The size of lexicon is different of number of vectors")

        fVec.close()

        return lexicon, Embedding(lexicon, vectors)