#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import sys
from wnn_ofertas import OfertasReader
from data.BatchIterator import AsyncBatchIterator
from data.Lexicon import createLexiconUsingFile


def genClassDistribution(filename, lexicon, norm=False):
    # Function to update the counts dict and the total counter.
    def encodeLabel(label):
        lex = lexicon.put(label)
        if lex < 0:
            sys.stderr.write("Label %s does not exist!" % str(label))
            sys.exit(1)
        return lex

    # Iterator that already update the counts dictionary.
    with AsyncBatchIterator(OfertasReader(filename), [], [encodeLabel], -1, shuffle=False, maxqSize=1000) as iterator:
        # Read labels from all examples.
        total = 0
        counts = {}
        for _, label in iterator:
            label = lexicon.getLexicon(label[0].item())
            c = counts.get(label, 0)
            counts[label] = c + 1
            total += 1
            if total % 100000 == 0:
                sys.stderr.write(".")
                sys.stderr.flush()
        sys.stderr.write(" done!\n")

    # Normalize the counts so that counts becomes a real distribution.
    if norm:
        for k, v in counts.iteritems():
            counts[k] = float(v) / total

    return counts


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("Syntax error! Expected syntax:\n")
        sys.stderr.write("\t<input file> <lexicon file> [--norm]\n")
        sys.exit(1)

    norm = False
    if len(sys.argv) == 4:
        if sys.argv[3] == "--norm":
            norm = True
        else:
            sys.stderr.write("Unexpected argument %s" % sys.argv[3])
            sys.exit(1)

    lexicon = createLexiconUsingFile(sys.argv[2])
    json.dumps(genClassDistribution(sys.argv[1], lexicon, norm))
