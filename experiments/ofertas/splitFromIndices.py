#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 12/07/2016

@author: eraldo

A partir de um dataset de ofertas, gera um dataset de treino e outro de teste.
Os índices dos exemplos de teste são fornecidos em um arquivo de entrada (um
índice em cada linha) e os demais exemplos são considerados de treino.

"""
import sys
from codecs import open


# Input file includes a header?
hasHeader = False

def docsFile2Dir(inputFilename, testIndicesFilename):
    print 'Reading test indices...'
    testIndicesFile = open(testIndicesFilename, 'r', 'utf8')
    testIndices = set()
    for l in testIndicesFile:
        testIndices.add(int(l))
    testIndicesFile.close()
    print '# test examples:', len(testIndices)

    # Open input dataset file.
    inFile = open(inputFilename, 'r', 'utf8')

    # Read header.
    if hasHeader:
        header = inFile.readline()

    # Output files: train and test.
    fTrain = open(inputFilename + '.train', 'w', 'utf8')
    fTest = open(inputFilename + '.test', 'w', 'utf8')

    # Write header.
    if hasHeader:
        fTrain.write(header)
        fTest.write(header)

    idx = 0
    numTestExs = 0
    numTrainExs = 0
    for l in inFile:
        if idx in testIndices:
            fTest.write(l)
            numTestExs += 1
        else:
            fTrain.write(l)
            numTrainExs += 1
        idx += 1

    numExs = idx

    # Verify range of test indices.
    for idx in testIndices:
        if idx < 0 or idx >= numExs:
            print "Error! Index %d is invalid!" % idx

    print '# input examples:', numExs
    print '# test  examples:', numTestExs
    print '# train examples:', numTrainExs

    # Close files.
    inFile.close()
    fTest.close()
    fTrain.close()

    print 'Done!'


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('Erro de sintaxe. Sintaxe esperada:\n')
        sys.stderr.write('\t<dataset de entrada> <arquivo de indices do teste>\n')
        sys.exit(1)

    docsFile2Dir(sys.argv[1], sys.argv[2])
