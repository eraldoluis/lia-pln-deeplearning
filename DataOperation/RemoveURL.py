#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

class RemoveURL:
    
    def filter(self,token):
        if re.search('.com', token) or re.search('http://', token):
            return '<url>'
        return token        
        
    