#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
import codecs
from wnn_ofertas import OfertasReader
from data.BatchIterator import AsyncBatchIterator
from data.Lexicon import createLexiconUsingFile


def genFeatureLexicon(inFilename, outFilename, feature):
    # Iterator that already update the counts dictionary.
    with OfertasReader(inFilename) as iterator, \
            codecs.open(filename=outFilename, mode="wt", encoding="utf8") as outFile:
        values = set()
        total = 0
        for offer, _ in iterator.read():
            ftrVal = offer[feature].strip().lower()
            if len(ftrVal) > 0 and ftrVal not in values:
                values.add(ftrVal)
                outFile.write(ftrVal + "\n")
            total += 1
            if total % 100000 == 0:
                sys.stderr.write(".")
                sys.stderr.flush()
        sys.stderr.write(" done!\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Syntax error! Expected syntax:\n")
        sys.stderr.write("\t<input file> <output file> <feature name>\n")
        sys.exit(1)

    genFeatureLexicon(inFilename=sys.argv[1], outFilename=sys.argv[2], feature=sys.argv[3])
