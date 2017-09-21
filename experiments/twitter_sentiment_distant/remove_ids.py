#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
from codecs import open
from sys import stdout
if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Missing Argument: <input file>."
		exit(1)
	infile = open(sys.argv[1], 'r')
	lines = infile.readlines()
	outfile = open(sys.argv[1], 'w')
	for line in lines :
		p = line.split('\t')
		outfile.write(p[1])
		
