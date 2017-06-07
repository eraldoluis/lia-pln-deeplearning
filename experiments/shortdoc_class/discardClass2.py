#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    numWrittenExs = 0
    print 'Reading input examples...'
    for l in inFile:
        (txt, cls) = l.split('\t')
        txt = txt.strip()
        cls = cls.strip()
        if cls != "2":
            outFile.write(u"{0}\t{1}\n".format(txt, cls))
            numWrittenExs += 1

        numExs += 1

    inFile.close()
    outFile.close()

    sys.stderr.write('\n')
    sys.stderr.write('# examples: %d\n' % numExs)
    sys.stderr.write('# written examples: %d\n' % numWrittenExs)
    sys.stderr.write('Done!\n')
