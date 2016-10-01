#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/07/2016

@author: eraldo

A partir de um dataset de ofertas, gera um dataset de treino e outro de 
desenvolvimento. Os exemplos são sorteados aleatoriamente, de acordo com a 
probabilidade fornecida na linha de comando.

"""
import random
import sys
from codecs import open


def docsFile2Dir(inputFilename, devProb):
    # Open input dataset file.
    inFile = open(inputFilename, 'r', 'utf8')
    header = inFile.readline()

    # Output files: train and test.
    fTrain = open(inputFilename + '.train', 'w', 'utf8')
    fDev = open(inputFilename + '.dev', 'w', 'utf8')

    # Write header.    
    fTrain.write(header)
    fDev.write(header)

    idx = 0
    numDevExs = 0
    numTrainExs = 0
    for l in inFile:
        if random.random() <= devProb:
            fDev.write(l)
            numDevExs += 1
        else:
            fTrain.write(l)
            numTrainExs += 1
        idx += 1

    print '# input examples:', idx
    print '# train examples:', numTrainExs
    print '# dev   examples:', numDevExs

    # Close files.
    inFile.close()
    fDev.close()
    fTrain.close()

    print 'Done!'


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.stderr.write('Erro de sintaxe. Sintaxe esperada:\n')
        sys.stderr.write('\t<dataset de entrada> <probabilidade do dev>\n')
        sys.exit(1)

    docsFile2Dir(sys.argv[1], float(sys.argv[2]))
