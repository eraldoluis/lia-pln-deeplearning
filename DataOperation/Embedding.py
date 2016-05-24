#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Classes and functions that operate on distributed representations
'''


import numpy
import logging
from NNet.Util import FeatureVectorsGenerator
import codecs
from DataOperation.Lexicon import Lexicon


######################################################
# Strategy to Generate the embedding, which represents the unknown object, randomly
##################################################

class UnknownGenerateStrategy:
    '''
    Abstract class
    '''
        
    unknownNameDefault = u'UUUNKKK'
    
    def getUnknownStr(self):
        return UnknownGenerateStrategy.unknownNameDefault
    
    def generateUnkown(self,embedding):
        '''
        :type embedding: Embedding
        '''
        pass
    
class RandomUnknownStrategy(UnknownGenerateStrategy):
    '''
    Randomly Generate the embedding which represents the unknown object     
    '''
    def generateUnkown(self,embedding):        
        return FeatureVectorsGenerator().generateVector(embedding.getEmbeddingSize())

     
class ChosenUnknownStrategy(UnknownGenerateStrategy):
    '''
    Get a object from lexicon to represent the unknown objects
    '''    
    def __init__(self,unknownName):
        self.__unknownName= unknownName
        self.__randomUnknownStrategy = RandomUnknownStrategy()
    
    def getUnknownStr(self):
        return self.__unknownName
    
    def generateUnkown(self,embedding):
        if embedding.exist(self.__unknownName):
            return embedding.get(self.__unknownName)
        
        return self.__randomUnknownStrategy.generateUnkown(embedding)


#####################################################################
# EmbeddingFactory
####################################################################

class EmbeddingFactory(object):
    '''
    Create embeddings
    '''

    def  __init__(self):
        self.__log = logging.getLogger(__name__)
    
    def createFromW2V(self,w2vFile):
        '''
        Create a embedding from word2vec output
        '''
        fVec = codecs.open(w2vFile, 'r','utf-8')
        
        # Read the number of words in the dictionary and the embedding size
        nmWords, embeddingSizeStr = fVec.readline().strip().split(" ")
        embeddingSize = int(embeddingSizeStr)
        
        embedding = Embedding(embeddingSize,RandomUnknownStrategy())
        
        for line in fVec:
            splitLine =  line.rstrip().split(u' ')

            word = splitLine[0]

            if len(word) == 0:
                self.__log.warning("Insert in the embedding a empty string")

            vec = [float(num) for num in splitLine[1:]]

            embedding.put(word, vec)
            
        embedding.stopAdd()
        fVec.close()
        
    
        return embedding
    
    def createEmptyEmbedding(self,embeddingSize):
        '''
        Create a embedding which give for each object a random vector
        '''
        return RandomEmbedding(embeddingSize,RandomUnknownStrategy())
        
        
        
    
    
#####################################################################
# Embedding classes
####################################################################

class Embedding(object):
    '''
    Represent an object distributed representation.
    This class has a matrix with all vectors and lexicon.
    '''
    
    def __init__(self,embeddingSize,unknownGenerateStrategy):
        """
        :type embeddingSize: int
        :params embeddingSize: the vectors length that represent the objects
        
        :type unknownGenerateStrategy: UnknownGenerateStrategy
        :params unknownGenerateStrategy: the object that will generate the unknown embedding
        
        """
        
        self.__lexicon = Lexicon()        
        self.__vectors = []
        self.__embeddingSize = embeddingSize
        self.__unknownGenerateStrategy = unknownGenerateStrategy
        
        # Stop to add new objects
        self.__stopAdd = False
            
    def stopAdd(self):
        '''        
        Stop  to add new objects and generate the unknown embedding
        '''
        if self.isStopped():
            return
        
        Embedding.put(self,self.__unknownGenerateStrategy.getUnknownStr(),
                    self.__unknownGenerateStrategy.generateUnkown(self))
        self.__lexicon.setUnknownIndex(self.getLexiconIndex(self.__unknownGenerateStrategy.getUnknownStr()))
        
        
        self.__stopAdd = True
        
    def isStopped(self):
        '''
        return if the class is not adding more new objects
        '''
        return self.__stopAdd
    
    def put(self,obj,vec=None):
        """
        :type obj:str
        :params obj: object to be added
        
        :type vec: list of double
        :params vec: vector which represents obj
        
        :return embedding of the object
        
        Add a new object to the embedding. 
        If the attribute stopAdd is False, vec is not none or object exists in lexicon, so the object index is returned.
        """
        if vec is None or self.isStopped() or self.__lexicon.exist(obj):
            return self.getLexiconIndex(obj)

        if len(vec) != self.__embeddingSize:
            raise Exception("the added vector has a different size of " + str(self.__embeddingSize))

        idx = self.__lexicon.put(obj)
        
        if len(self.__vectors) != idx:
            raise Exception("Exist more or less lexicon than vectors")
            
        self.__vectors.append(vec)
        
        return idx
    
    def exist(self,obj):
        return self.__lexicon.exist(obj)
    
    def getLexiconIndex(self,obj):
        return self.__lexicon.getLexiconIndex(obj)

    def getEmbeddingByIndex(self,idx):
        return self.__vectors[idx]
    
    def getEmbedding(self,obj):
        idx = self.__lexicon.getLexiconIndex(obj)
        return self.getEmbeddingByIndex(idx)
    
    def getEmbeddingMatrix(self):
        return self.__vectors
    
    def getNumberOfEmbeddings(self):
        return len(self.__vectors)
            
    def getEmbeddingSize(self):
        return self.__embeddingSize
    
    def getLexicon(self):
        '''
        :return data_operation.lexicon.Lexicon
        '''
        return self.__lexicon
    
        
class RandomEmbedding(Embedding):
    '''
    In this embedding each new added object  receive a random vector
    '''
        
    def __init__(self,embeddingSize,unknownGenerateStrategy):
        Embedding.__init__(self,embeddingSize,unknownGenerateStrategy)
        
        # Generator that going to generate values for vectors 
        self.__generatorWeight = FeatureVectorsGenerator()
        

    def put(self, obj):
        vec = self.__generatorWeight.generateVector(self.getEmbeddingSize())
        return Embedding.put(self, obj, vec)
        