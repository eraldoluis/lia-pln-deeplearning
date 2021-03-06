#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 26/05/2017

@author: eraldo

A partir de um arquivo contendo tweets (um em cada linha e com possíveis atributos separados por TAB),
aplica os passos abaixo e salva em outro arquivo.
    - Tokenização.
    - Converte tudo para minúsculo.
    - Substitui dígitos por 0.
    - Substitui tokens iniciados em 'http:' e 'www.' por ##LINK##.
    - Substitui hashtags por ##HASHTAG##.
    - Substitui @xxx por ##REF##.
    - Substitui repetições de sinais de pontuação por um único sinal.
    - Substitui sequências de pontos por ...
    - Substitui sequências de exclamações por !!!

    Nota: Esse script ignora a primeira linha (head) do arquivo "input" 
    
    Nota: Script modificado para manter hashtags
"""
import re
import sys
from codecs import open

from experiments.shortdoc_class.tokenizer import getTokenizer

import unicodedata

from datetime import datetime
def normalize(word):
    normalizedWord = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore')

    if(len(normalizedWord) != 0):
        return normalizedWord
    else:
        return word

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Syntax error!"
        print "\tArguments: <input file> <output file>"
        print "Both input and output files can be -, which means standard input or output, respectively."
        sys.exit(1)

    inFilename = sys.argv[1]

    inFile = sys.stdin
    if inFilename != "-":
        inFile = open(inFilename, "rt", "utf-8")

    outFilename = sys.argv[2]
    outFile = sys.stdout
    if outFilename != "-":
        outFile = open(outFilename, "wt", "utf-8")

    tokenizer = getTokenizer()

    # Skip header line.
    inFile.readline()
    numExs = 0
    print(str(datetime.now().time()) + "- Reading input examples...")
    # Recognize digits.
    patDig = re.compile('[0-9]')
    # Recognize sequence of punctuations.
    patPunc = re.compile('([!.,?])+')
    for l in inFile:
        # Each line can be composed by several features.
        ftrs = l.split('\t')

        # Clean possible \n in the end of the line.
        ftrs[-1] = ftrs[-1].strip()

        # Tokenize the text.
        textIndex = 1
        tokens = tokenizer.tokenize(ftrs[textIndex])

        # Apply filters.
        procTokens = []
        for token in tokens:
            if token.startswith('http:') or token.startswith('https:') or token.startswith('www.'):
                token = "__LINK__"
            elif token.startswith("@"):
                token = "__REF__"
            elif token.startswith(".."):
                token = "..."
            elif token.startswith("!!"):
                token = "!!!"
            else:
                if not token.startswith("#"):
                    token = re.sub(patDig, '0', token)
                token = token.lower()
                token = re.sub(patPunc, r'\1', token)

            normalizedToken = normalize(unicode(token))

            if(len(normalizedToken) > 0 and normalizedToken != "#"):
                token = normalizedToken

            procTokens.append(token)

        # If there isn't any token, line won't be written on the new file
        if (len(tokens) != 0):
            ftrs[1] = " ".join(procTokens)
            outFile.write("\t".join(ftrs) + "\n")
            numExs += 1

    inFile.close()
    outFile.close()

    sys.stderr.write('\n')
    sys.stderr.write('# examples: %d\n' % numExs)
    sys.stderr.write(str(datetime.now().time()) + ' - Done!\n')
