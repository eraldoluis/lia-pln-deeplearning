#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

class RemoveUserName:
    
    def filter(self,token):
        if re.search('@[_]*[a-zA-Z0-9]+', token):
            return u'<user>'
    
        return token       
                
        
    