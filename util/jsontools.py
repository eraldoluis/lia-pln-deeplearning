#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 24 de set de 2016

@author: eraldo
"""

import json
from collections import namedtuple
from logging import Formatter


def _json_object_hook(d, typename='T'):
    return namedtuple(typename, d.keys())(*d.values())


def jsonstr2obj(s):
    return json.loads(s, object_hook=_json_object_hook)


def dict2obj(mydict, typename='T'):
    for k, v in mydict.items():
        if isinstance(v, dict):
            mydict[k] = dict2obj(v, typename + k.capitalize())

    return namedtuple(typename, mydict.keys())(*mydict.values())


class JsonLogFormatter(Formatter):
    """
    Format messages to be included in JSON strings. It just pass the msg attribute of the LogRecord through
    json.dumps(msg).
    """

    def __init__(self, fmt=None, datefmt=None):
        if not fmt:
            fmt = '{"timestamp": "%(asctime)s", "module": "%(name)s", "level": "%(levelname)s", "message": %(message)s}'
        if not datefmt:
            datefmt = '%Y/%m/%d %H:%M:%S'
        super(JsonLogFormatter, self).__init__(fmt, datefmt)

    def format(self, record):
        record.msg = json.dumps(record.msg)
        return super(JsonLogFormatter, self).format(record)