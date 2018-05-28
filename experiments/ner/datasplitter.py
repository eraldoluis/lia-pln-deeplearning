#!/usr/bin/env python
# -*- coding: utf-8 -*-
from random import random
from argparse import ArgumentParser

# Plotar LossTrain e CustomDev F value para epoca
# Plotar para 15 épocas primeiro

def split(filename, bias = 0.5):
	"""
	Splits a dataset file into two using a biased randomic selection.
	:param filename: File to be split.
	:param bias: Biasing value.
	"""
	selectfile = open(filename + "_train.txt", "w")
	rejectfile = open(filename + "_dev.txt", "w")
	targetfile = (selectfile, rejectfile)
	counter = [0, 0]

	with open(filename, 'r') as original:
		selection = (random() <= bias)
		destiny = targetfile[selection]
		counter[selection] += 1
		
		line = original.readline()
		
		while line:

			if line[0] is '\n':
				continue
		
			destiny.write(line)
			
			if line[0] in ['.', '!', '?']:
				selection = (random() <= bias)
				destiny = targetfile[selection]
				counter[selection] += 1
			
			line = original.readline()
			
	print counter
			
if __name__ == "__main__":
	
	# Creation of the command line argument parser. This will allow
	# the script to receive many different arguments from command line.
	parser = ArgumentParser(
		description = "Splits a dataset into two different ones, "
		              "allowing random and/or biased selections."
	)
	
	parser.add_argument(
		'file',
		nargs = 1,
		type = str,
		help = "Dataset file to be split into two."
	)
	
	parser.add_argument(
		'--bias', '-b',
		nargs = 1,
		type = float,
		default = [0.5],
		help = "Bias value to affect data destination."
	)
	
	args = parser.parse_args()
	split(args.file[0], args.bias[0])