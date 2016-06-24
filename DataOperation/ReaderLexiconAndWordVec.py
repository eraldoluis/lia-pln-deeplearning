#!/usr/bin/env python
# -*- coding: utf-8 -*-

from DataOperation.Lexicon import Lexicon
from DataOperation.WordVector import WordVector
import re
import codecs


class ReaderLexiconAndWordVec:    
   
    def readData(self, filename):
        '''
        Read the data from a file and return two structures: Lexicon (containing de dict of words) and wordVector (containing the matrix with word representations)
        '''
        lexicon = Lexicon()
        wordVector = WordVector()
        
        f = codecs.open(filename, 'r','utf-8')
        a = 0
        
        for line in f:
            
            if re.search('^[0-9]+ [0-9]+$', line):
                if a > 0:
                    raise Exception('Foi encontrado mais de uma linha no arquivo que contém o número de exemplos e tamanho do word vector')
                a+=1
                continue;
            
            
            line_split = line.split(' ', 1 );
            
            if len(line_split) < 2:
                continue 
            
            idx = lexicon.getLexiconIndex(line_split[0].lower())
            if lexicon.isUnknownIndex(idx):
                lexicon.put(line_split[0].lower());
                wordVector.putWordVecStr(line_split[1])
            
            
        f.close()
        
        return [lexicon,wordVector]
    
    def simpleRead(self, filename):
        
        labelDict = {}
        labels = [] 
        lines = []
        
        f = codecs.open(filename, 'r','utf-8')
        
        for line in f:
            
            if len(line) < 1:
                continue
                
            line_split = line.split(',', 1 );
                
            if len(line_split) < 2:
                    continue 
            
            idxLabel = labelDict.get(line_split[0], -1)
            if idxLabel == -1:
                labelDict[line_split[0]] = len(labelDict)
                idxLabel = labelDict[line_split[0]]
                labels.append(line_split[0])
                lines.append([])
                        
            lines[idxLabel].append(line_split[1])  
        
        f.close()
        
        return [labels, lines]

    