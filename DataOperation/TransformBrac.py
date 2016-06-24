#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

class TransformBrac:
    
    def filter(self,token):
                   
        token = re.sub('-LRB-', '(', token)
        token = re.sub('-RRB-', ')', token)
        
        token = re.sub('-LSB-', '[', token)
        token = re.sub('-RSB-', ']', token)
        
        token = re.sub('-LCB-', '{', token)
        token = re.sub('-RCB-', '}', token)
                   
        return token        
        
    