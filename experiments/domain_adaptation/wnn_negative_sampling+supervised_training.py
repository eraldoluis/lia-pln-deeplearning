#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script trains the unsupervised part using the negative_sampling
and, after this, it trains in a supervised way a network with
supervised and unsupervised part.
"""
import codecs
import json
import os
import logging.config

import sys

import experiments.domain_adaptation.wnn_negative_sampling as ns
import experiments.domain_adaptation.wnn_with_target_feature_module as wt
from args.JsonArgParser import JsonArgParser

from util.jsontools import dict2obj

PARAMETERS = {
    "negative_sampling_json_base": {
        "required": True, "desc": "a file with a json with the default parameter values  of the negative sampling"
    },
    "wnn_with_target_json_base": {
        "required": True, "desc": "a file with a json with the default parameter values  of final neural network "
    },
    "lr_ns": {
        "required": True, "desc": "learning rate of negative sampling"
    },
    "lr_wnn": {
        "required": True, "desc": "learning rate of the final neural network"
    },
    "noise_rate": {"required": True, "desc": "Number of noise examples",},

    "num_epochs": {"required": True, "desc": "Number of epochs in negative sampling"},
    "t": {"required": True,
          "desc": "Set threshold for occurrence of words. Those that appear with higher frequency in the training data "
                  "will be randomly down-sampled; default is 1e-5, useful range is (0, 1e-10)"},

    "power": {"required": True, "desc": "q(w)^power, where q(w) is the unigram distribution."},
    "min_count": {"required": True,
                  "desc": "This will discard words that appear less than n times",},
    "save_model": {"required": True,
                  "desc": "File to save the model trained using negative sampling",},
}


def main(args):
    #
    jsonNs = JsonArgParser(ns.PARAMETERS).parse(args.negative_sampling_json_base)

    jsonNs["lr"] = args.lr_ns
    jsonNs["noise_rate"] = args.noise_rate
    jsonNs["num_epochs"] = args.num_epochs
    jsonNs["t"] = args.t
    jsonNs["power"] = args.power
    jsonNs["min_count"] = args.min_count
    jsonNs["save_model"] = args.save_model

    jsonWnn = JsonArgParser(wt.WNN_PARAMETERS).parse(args.wnn_with_target_json_base)
    jsonWnn["lr"] = args.lr_wnn
    jsonWnn["target_module"] = args.save_model


    log = logging.getLogger(__name__)

    log.info("Starting unsupervised training")
    ns.mainWnnNegativeSampling(dict2obj(jsonNs))

    log.info("Starting supervised training")
    wt.mainWnn(dict2obj(jsonWnn))


if __name__ == '__main__':
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))

    parameters = dict2obj(JsonArgParser(PARAMETERS).parse(sys.argv[1]))
    main(parameters)
