#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Filters are classes that change the token, like transform to lower case the letters 
'''

import re


class Filter:
    
    def filter(self,token):
        raise NotImplementedError()

class TransformLowerCaseFilter:
    
    def filter(self,token):        
        return token.lower()

class TransformNumberToZeroFilter:

    def filter(self,token):        
        return re.sub('[0-9]', '0', token)