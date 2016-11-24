#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import sys


def genLabelMapping(idLabelFilename, lexiconFilename, outFilename):
    idLabel = {}
    # Iterator that already update the counts dictionary.
    with codecs.open(idLabelFilename, "rt", "utf8") as idLabelFile:
        count = 0
        for l in idLabelFile:
            count += 1
            vals = [s.strip() for s in l.split("\t")]
            if len(vals) != 2:
                print 'Warning: line %d with less than 2 values! Ignoring.' % (count + 1)
                continue
            (id, label) = vals
            idLabel[id] = label
        print 'Read %d id-label pairs.' % count

    with codecs.open(lexiconFilename, "rt", "utf8") as lexiconFile, codecs.open(outFilename, "wt", "utf8") as outFile:
        count = 0
        for l in lexiconFile:
            count += 1
            l = l.strip()
            outFile.write(idLabel[l] + "\n")
        print "Written %d label descriptions." % count


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.stderr.write("Syntax error! Expected syntax:\n")
        sys.stderr.write("\t<id-label file> <label lexicon file> <output file>\n")
        sys.exit(1)

    genLabelMapping(sys.argv[1], sys.argv[2], sys.argv[3])
