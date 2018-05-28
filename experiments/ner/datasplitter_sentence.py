#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import argv
import sys

"""
Divide um arquivo em duas partes: train e dev.

Este programa recebe três argumentos: <file> <train_count> <dev_count>,
sendo que <file> é o arquivo contendo um token por linha e frases
separadas por uma linha em branco;
<train_count> indica quantas frases consecutivas serão redirecionadas
para o arquivo de treino;
e <dev_count> indica quantas frases consecutivas serão redirecionadas
para o arquivo de validação (dev).
O script percorre as frases do arquivo de entrada, redirecionando
<train_count> frases para o arquivo de treino e <dev_count> para
o arquivo de validação.
"""


def split(filename, trainStep, devStep):
    trainFile = open(filename + "_train.txt", "w")
    devFile = open(filename + "_dev.txt", "w")
    destFiles = (trainFile, devFile)
    steps = (trainStep, devStep)
    counts = [0, 0]
    tokenCounts = [0, 0]

    with open(filename, 'r') as inFile:
        dest = 0

        line = inFile.readline()
        while len(line) > 0:
            destFiles[dest].write(line)

            if line == '\n':
                counts[dest] += 1
                if (counts[dest] % steps[dest]) == 0:
                    dest = 1 - dest
            else:
                tokenCounts[dest] += 1

            line = inFile.readline()

        counts[dest] += 1

    print "Total sentences: %d" % (counts[0] + counts[1])
    print "Train sentences: %d" % counts[0]
    print "Dev   sentences: %d" % counts[1]
    print ""
    print "Total tokens: %d" % (tokenCounts[0] + tokenCounts[1])
    print "Train tokens: %d" % tokenCounts[0]
    print "Dev   tokens: %d" % tokenCounts[1]


if __name__ == "__main__":
    if len(argv) != 4:
        sys.stderr.write("Syntax error. Arguments expected: <filename> <train_count> <dev_count>\n")
        sys.exit(1)

    filename = argv[1]
    trainStep = int(argv[2])
    devStep = int(argv[3])

    split(filename, trainStep, devStep)
