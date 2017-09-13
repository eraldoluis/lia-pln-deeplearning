#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 21/09/2016

@author: eraldo

A partir de um dataset de ofertas, gera um arquivo de entrada para o utilitário
word2vec. Este arquivo contém uma oferta em cada linha e os tokens são separados
por espaço. Na realidade, o único pré-processamento feito por este script
consiste em:
    - obter somente o texto das ofertas (descartando os demais atributos);
    - subtituir dígitos por 0.
"""
import re
import sys
from codecs import open

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Syntax error!"
        print "\tArguments: <input file> <output file>"
        print "Both input and output files can be -, which means standard input or output, respectively."
        sys.exit(1)

    inFilename = sys.argv[1]
    inFile = sys.stdin
    if inFilename != "-":
        inFile = open(inFilename, "rt", "utf8")

    outFilename = sys.argv[2]
    outFile = sys.stdout
    if outFilename != "-":
        outFile = open(outFilename, "wt", "utf8")

    # Skip header line.
    inFile.readline()
    numExs = 0
    print 'Reading input examples...'
    pat = re.compile('[0-9]')
    for l in inFile:
        txt = l.split('\t')[0].lower()
        txt = re.sub(pat, '0', txt)
        outFile.write(txt + "\n")

        numExs += 1

    inFile.close()
    outFile.close()
    
    sys.stderr.write('\n')
    sys.stderr.write('# examples: %d\n' % numExs)
    sys.stderr.write('Done!\n')
