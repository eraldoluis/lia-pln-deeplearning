#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import codecs
import logging

log = logging.Logger(__name__)

def getPropertyFromDict(d, prop):
    """
    Return the value of a property given its qualified name (including dots, like "message.name"). If such property
    does not exist, return None.

    :param d: dictionary where to search for the property in.
    :param prop: qualified name of the property.
    :return: the value of the given property or None, if the property does not exist.
    """
    props = prop.split(".")
    for p in props:
        if not isinstance(d, dict):
            return None
        if p not in d:
            return None
        d = d[p]
    return d


def readSequence(filename, filter, properties):
    """
    Read and return a sequence of properties from a JSON log file.

    :param filename: name of a JSON log file.
    :param filter: a tuple (or sequence) with two values: filter[0] is the property name and filter[1] is the property
        value. Only entries whose given property is equal to the given value are considered.
    :param properties: list of properties whose values are appended to the output sequence.
    :return: a sequence of properties found in the given log file.
    """
    seq = []
    with codecs.open(filename, mode="r", encoding="utf8") as f:
        for l in f:
            try:
                d = json.loads(l)
                if filter[1] == getPropertyFromDict(d, filter[0]):
                    seq.append([getPropertyFromDict(d, prop) for prop in properties])
            except ValueError as e:
                log.error("Error loading JSON", e)
    return seq


# print readSequence(filename="/home/eraldo/log.teste.txt",
#                    filter=("message.name", "EvalFMetric"),
#                    properties=("message.epoch",
#                                "message.iteration",
#                                "message.values.macro.f",
#                                "message.values.micro.f"))
