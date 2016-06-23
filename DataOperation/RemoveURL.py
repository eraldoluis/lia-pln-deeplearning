#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

class RemoveURL:
    
    def filter(self,token):
        if re.search('http://[a-zA-Z]+', token.lower()) or re.search('https://[a-zA-Z]+', token.lower()):
            return u'<url>'
        return token        
        
    
