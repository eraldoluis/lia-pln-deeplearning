#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import json


class JsonArgParser:

    def __init__(self, parameterRule):
        '''

        :type parameterRule: basestring
        :param parameterRule: Its a json string which contains description and default values of some parameters.
            Besides that, it's possible specific if a parameter is required or not. If nothing is stipulate, so
            the parameter will be consider not required.
            Below is showed the json structure:
            For instance:
                {
                    "parameter_name1": {"default": 300 , "desc": "...."},
                    "parameter_name2": {"default": "..."},
                    "parameter_name3": {"required":True}
                }
        '''

        if isinstance(parameterRule, (basestring, unicode)):
            self.__parameterRules = json.loads(parameterRule)
        else:
            self.__parameterRules = parameterRule

    def parse(self, f):
        '''
        :type f: basestring or Stream
        :param f: Can be the output of codecs.open or open, or the path of a file.
        :return: a dictionary which contains the parsed parameters.
        '''

        if isinstance(f, basestring):
            f = codecs.open(f, "r", "utf-8")
            # f = open(f, "r")

        parameters = json.load(f)

        for paramName, rule in self.__parameterRules.iteritems():
            default = rule.get("default")
            parameterNotExist = parameters.get(paramName) is None

            if rule.get("required"):
                if parameterNotExist:
                    raise Exception(u"Parameter " + paramName + u" is required.")

            if parameterNotExist:
                parameters[paramName] = default


        return parameters
