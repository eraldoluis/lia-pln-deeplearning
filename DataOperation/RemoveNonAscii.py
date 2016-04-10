#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unicodedata

class RemoveNonAscii:
    
    def filter(self,token):
        #t = unicode(token, "utf-8")
        #return unicodedata.normalize('NFKD', t ).encode('ascii', 'ignore')
        return unicodedata.normalize('NFD', token.decode('unicode-escape', 'ignore')).encode('ascii', 'ignore')
        