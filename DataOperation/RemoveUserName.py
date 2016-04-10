#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

class RemoveUserName:
    
    def filter(self,token):
        if re.search('@[_a-zA-Z]+', token):
            return '<user>'
        return token        
        
    