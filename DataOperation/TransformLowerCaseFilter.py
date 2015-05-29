#!/usr/bin/env python
# -*- coding: utf-8 -*-

class TransformLowerCaseFilter:
    
    def filter(self,token):        
        return token.lower()
        