#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 24 de set de 2016

@author: eraldo
'''

import json
from collections import namedtuple

def _json_object_hook(d, typename='T'):
    return namedtuple(typename, d.keys())(*d.values())

def jsonstr2obj(s):
    return json.loads(s, object_hook=_json_object_hook)

def dict2obj(mydict, typename='T'):
    for k, v in mydict.items():
        if isinstance(v, dict):
            mydict[k] = dict2obj(v, typename + k.capitalize())
    return namedtuple(typename, mydict.keys())(*mydict.values())
