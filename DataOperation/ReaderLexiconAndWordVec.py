#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DataOperation.Lexicon import Lexicon
from DataOperation.WordVector import WordVector
import re


class ReaderLexiconAndWordVec:    
   
    def readData(self, filename):
        '''
        Read the data from a file and return a matrix which the first row is the words indexes  and second row is the labels values
        '''
        lexicon = Lexicon()
        wordVector = WordVector()
        
        f = open(filename, 'r')
        a= 0
        
        for line in f:
            
            if re.search('^[0-9]+ [0-9]+$', line):
                if a > 0:
                    raise Exception('Foi encontrado mais de uma linha que contém o número de exemplos e tamanho do word vector')
                a+=1
                continue
            
            
            line_split = line.split(' ', 1 );
            
            lexicon.put(line_split[0]);
            wordVector.putWordVecStr(line_split[1])
            
            
        f.close()
        
        return [lexicon,wordVector]
 