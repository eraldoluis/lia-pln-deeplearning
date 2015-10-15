#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unicodedata

class RemoveAccentFilter:
    
    def filter(self,token):
        #t = unicode(token, "utf-8")
        #return unicodedata.normalize('NFKD', t ).encode('ascii', 'ignore')
        return unicodedata.normalize('NFKD', token ).encode('ascii', 'ignore')
        