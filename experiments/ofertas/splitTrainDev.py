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


def docsFile2Dir(inputFilename, devProb, suffix):
    # Open input dataset file.
    inFile = open(inputFilename, "r", "utf8")
    header = inFile.readline()

    # Output files: train and test.
    fTrain = open(inputFilename + suffix + ".train", "w", "utf8")
    fDev = open(inputFilename + suffix + ".dev", "w", "utf8")

    print "Train file:", fTrain.name
    print "Dev.  file:", fDev.name

    # Write header.
    fTrain.write(header)
    fDev.write(header)

    sys.stdout.write("Generating")
    numExs = 0
    numDevExs = 0
    numTrainExs = 0
    for l in inFile:
        if random.random() <= devProb:
            fDev.write(l)
            numDevExs += 1
        else:
            fTrain.write(l)
            numTrainExs += 1
        numExs += 1
        if numExs % 100000 == 0:
            sys.stdout.write(".")
            sys.stdout.flush()

    sys.stdout.write(" done!\n")

    print "# input examples:", numExs
    print "# train examples:", numTrainExs
    print "# dev   examples:", numDevExs

    # Close files.
    inFile.close()
    fDev.close()
    fTrain.close()

    print "Done!"


if __name__ == "__main__":
    if len(sys.argv) not in (3,4):
        sys.stderr.write("Erro de sintaxe. Sintaxe esperada:\n")
        sys.stderr.write("\t<dataset de entrada> <probabilidade do dev> [suffix]\n")
        sys.exit(1)

    suffix = ""
    if len(sys.argv) == 4:
        suffix = sys.argv[3]
    docsFile2Dir(sys.argv[1], float(sys.argv[2]), suffix)
