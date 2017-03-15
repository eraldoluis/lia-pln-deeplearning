#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib
import logging.config
import os
import subprocess

import theano.tensor as T


def getTheanoTypeByDimension(dim, name=None, dtype=None):
    if dim== 1:
        _input = T.vector(name, dtype)
    elif dim == 2:
        _input = T.matrix(name, dtype)
    elif dim == 3:
        _input = T.tensor3(name, dtype)
    elif dim == 4:
        _input = T.tensor4(name, dtype)
    else:
        raise Exception("Invalid dimension")

    return _input


def execProcess(cmd, logger, working_director=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=working_director)

    while p.poll() is None:
        l = p.stdout.readline()  # This blocks until it receives a newline.
        print l.rstrip()

    print p.stdout.read().rstrip()


def unicodeToSrt(s):
    return str(s.encode('utf-8'))


def getFileNameInPath(str):
    return os.path.split(str)[1]


def removeExtension(file):
    return os.path.splitext(file)[0]

def isUpper(word):
    """
    This method is used when the word encoding is unicode.
    :param word: unicode
    :return:
    """
    return sum(c.isupper() for c in word) == len(word)


def loadConfigLogging():
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)

    logging.config.fileConfig(os.path.join(path, 'logging.conf'))


def getFilters(filterNames, log):
    """
    Load and return all the filters in "filterNames" list.
    :param filterNames: a list with the filter names.
    :param log: log object.
    :return: the filters
    """
    filters = []

    for filterName in filterNames:
        moduleName, className = filterName.rsplit('.', 1)
        log.info("Usando o filtro: " + moduleName + " " + className)

        module_ = importlib.import_module(moduleName)
        filters.append(getattr(module_, className)())

    return filters

