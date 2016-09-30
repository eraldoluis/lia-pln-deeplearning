#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from itertools import chain
from timeit import itertools
from numpy import dtype, ndarray
import theano
from theano import tensor as T
from operator import itemgetter
from EmbeddingConvolutionalLayer import *
import numpy as np
import theano
import theano.tensor as T
from nnet.HiddenLayer import HiddenLayer
from nnet.EmbeddingLayer import EmbeddingLayer
from theano.tensor.nnet.nnet import softmax
from nnet.SoftmaxLayer import SoftmaxLayer
from nnet.Util import negative_log_likelihood, regularizationSquareSumParamaters



 
# O tamanho do batch
batchSize = theano.shared(1,'batchSize')

# O tamanho do vetor de caracter
sizeCharVector = 2

# O tamanho da janela dos caracteres
charWindowSize = 3

# O tamanho da janela das palavras
wordWindowSize = 2
wordWindowSizeT = theano.shared(wordWindowSize,'wordWindowSizeT')

# Criando char vector
cv = numpy.ones((20, sizeCharVector)) 
cv = numpy.array([	[1,2],
					[2,3],
					[3,4],
					[4,5],
					[5,6],
					[6,7],
					[7,8],
					[8,9],
					[9,10],
					[10,11],
					[11,12],
					[12,13],
					[13,14],
					[14,15],
					[0,9],
					[9,7],
					[2,4],
					[4,6],
					[3,4],
					[7,5]])
cvT = theano.shared(cv,'cv')

# Criando os pesos do hidden layer
hiddenLayerSize = 3
#WH = numpy.random.random_sample((charWindowSize * sizeCharVector, hiddenLayerSize))
WH = numpy.array([	[0,1,2],
				   	[1,2,3],
				   	[2,3,4],
				   	[3,4,5],
				   	[4,5,6],
				   	[5,6,1]])
#b = numpy.random.random_sample(hiddenLayerSize)
#[[2 3 3 4 4 5],[2 3 5 6 6 7],[5 6 6 7 4 5]]

# O WindowChar terá o índice das janelas dos caracteres de todas as palavras da janela.
# Por exemplo: 
#            janela de palvras = [o , cão, morreu]
#            windowChar = [
#                            [\,o,/],
#                            [\,c,ã]
#                            [c,ã,o]
#                            [ã,o,/]
#                            [\,m,o]
#                            [m,o,r]
#                            [o,r,r]...
#                            ]
windowChar = numpy.asarray([
                                            [1,2,3], #o
                                            [1,4,5], #ca
                                            [4,5,3],#cao
                                            [7,8,9],#mor
                                            [9,10,11],#rre
                                            [11,12,13],#reu
                                            [14,15,16]      
                                            ])

# Este vetor guardar número de caracteres por palavra
# Por exemplo: 
#            na windowChar do exemplo acima teremos
#            numCharByWord = [
#                            1,
#                            3,
#                            6,
#                            ]
numCharByWord = theano.shared(numpy.asarray([1,2,3,1]),'numCharByWord',int)

windowIdxs = theano.shared(value=np.zeros((1,charWindowSize),dtype="int64"),
                                   name="windowIdxs")
## Começo das operações 
cvFlatten = theano.printing.Print()(T.flatten(cvT[windowIdxs],2))

dot = theano.printing.Print()(T.dot(cvFlatten, WH))

# O número de palavras 
numWor = batchSize *  wordWindowSizeT

curIdx = T.scalar('curIdx',dtype='int64') 
wordIdx = T.scalar('wordIdx',dtype='int64') 

bb = T.zeros((numWor,dot.shape[1]))

indice = theano.shared(value=0,name="indice") 

# A função retorna o max de cada palavra
def maxByWord(ind,wordIndex,bb,indice,dot):
    numChar = numCharByWord[wordIndex]
    bb =  T.set_subtensor(bb[ind], T.max(dot[indice:indice+numChar], 0))
    #curIndex = curIndex + numChar
    wordIndex = wordIndex +1
    indice = indice + numChar
    return [wordIndex,bb,indice]


a = T.arange(0,numWor)
#print a.eval()
[i,r,s], updates = theano.scan(fn= maxByWord,
                   sequences = a,
                   outputs_info = [wordIdx,bb,indice],
                   non_sequences = dot,
                   n_steps = numWor)

# Este o vector que será concatenato com o do wordVector
ans = r[-1].reshape((batchSize,wordWindowSizeT * dot.shape[1]))
pos = i[-1]

windowIdxs.set_value(windowChar,borrow=True)

f = theano.function(inputs=[wordIdx], outputs=[pos,ans],
				givens={windowIdxs:windowIdxs[T.sum(numCharByWord[0:wordIdx]): T.sum(numCharByWord[0:wordIdx+wordWindowSizeT])]})

for j in range(3):
	print f(j);

