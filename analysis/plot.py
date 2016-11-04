#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import codecs


def getPropertyFromDict(d, prop):
    props = prop.split(".")
    for p in props:
        if not isinstance(d, dict):
            return None
        if p not in d:
            return None
        d = d[p]
    return d


logs = [
    "/home/eraldo/log.teste.txt"
]

filters = [
    {
        "operator": "and",
        "operands": {
            "message.type": "metric",
            "message.subtype": "evaluation",
            "message.name": "EvalFMetric"
        }
    }
]

with codecs.open(logs[0], mode="r", encoding="utf8") as f:
    for l in f:
        d = json.loads(l)
        if "metric" == getPropertyFromDict(d, "message.type") and \
                        "evaluation" == getPropertyFromDict(d, "message.subtype") and \
                        "EvalAccuracy" == getPropertyFromDict(d, "message.name"):
            print getPropertyFromDict(d, "message.epoch"), \
                getPropertyFromDict(d, "message.iteration"), \
                getPropertyFromDict(d, "message.values.accuracy")
