#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from json import JSONEncoder

if __name__ == "__main__":
	parser = ArgumentParser(
		description = "Creates a JSON to be used as parameter for CharWNN tests."
	)
	
	parser.add_argument(
		'directory',
		nargs = 1,
		type = str,
		help = "Directory in which data can be found"
	)

	parser.add_argument(
		'--lr',
		nargs = 1,
		type = float,
		default = [0.01],
	)
	
	parser.add_argument(
		'--batchsize',
		nargs = 1,
		type = int,
		default = [8],
	)
	
	parser.add_argument(
		'--decay',
		nargs = 1,
		type = str,
		default = ['DIVIDE_EPOCH']
	)
	
	parser.add_argument(
		'--numepochs',
		nargs = 1,
		type = int,
		default = [15]
	)
	
	parser.add_argument(
		'--shuffle',
		nargs = 1,
		type = bool,
		default = [True]
	)
	
	parser.add_argument(
		'--normalization',
		nargs = 1,
		type = str,
		default = ["minmax"]
	)
	
	parser.add_argument(
		'--seed',
		nargs = 1,
		type = int,
		default = [1]
	)
	
	args = parser.parse_args()
	directory = args.directory[0]
	
	if directory[-1] == '/':
		directory = directory[:-1]
	
	data = {
		"word_filters": ["data.Filters.TransformLowerCaseFilter", "data.Filters.TransformNumberToZeroFilter"],
		"label_file": directory + "/labels.txt",
		"train": directory + "/harem/first.txt_train.txt",
		"dev": directory + "/harem/first.txt_dev.txt",
		"test": directory + "/test_set_2.txt",
		"word_embedding": directory + "/ptwiki_cetenfolha_cetempublico.trunk.voc5.100wv.ctw5.sample1e_1.5k_vectors",
		"word_lexicon": directory + "/word_lexicon.txt",
		"char_lexicon": directory + "/char_lexicon.txt",
		"lr": args.lr[0],
		"batch_size": args.batchsize[0],
		"decay": args.decay[0],
		"num_epochs": args.numepochs[0],
		"shuffle": args.shuffle[0],
		"normalization": args.normalization[0],
		"seed": args.seed[0],
	}

	json = JSONEncoder()
	print json.encode(data)
