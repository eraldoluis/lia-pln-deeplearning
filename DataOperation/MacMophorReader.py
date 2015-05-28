#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
'''
import re
import codecs

class MacMorphoReader:
    
    def __init__(self,fileWithFeatures):
        self.fileWithFeatures = fileWithFeatures 
     
    def readTestData(self, filename,lexicon,lexiconOfLabel,separateSentences=True,filters=[]):
        return self.readData(filename, lexicon, lexiconOfLabel,None,separateSentences,False,filters)
    
    def addToken(self, lexicon, wordVecs, addWordUnkown, filters, indexesBySentence, word,lexiconFindInTrain):
        for f in filters:
            word = f.filter(word)
            
        lexiconIndex = lexicon.getLexiconIndex(word)
        if addWordUnkown and lexicon.isUnknownIndex(lexiconIndex):
            lexiconIndex = lexicon.put(word)
            wordVecs.append(None)
        
        if lexiconFindInTrain is not None:
            lexiconFindInTrain.add(word.lower())
        
        indexesBySentence.append(lexiconIndex)

    def addLabel(self, lexiconOfLabel, labelsBySentence, token):
        labelsBySentence.append(lexiconOfLabel.put(token))

    def readTokenAndLabelOfFileWithFeature(self, lexicon, lexiconOfLabel, wordVecs, addWordUnkown, filters, indexesBySentence, labelsBySentence, token,lexiconFindInTrain):
        prefWord = 'word='
        
        if prefWord in token:
            word = token[len(prefWord):]
            self.addToken(lexicon, wordVecs, addWordUnkown, filters, indexesBySentence, word,lexiconFindInTrain)
        elif re.search(r'^([A-Z]|\W)', token) is not None:
            self.addLabel(lexiconOfLabel, labelsBySentence, token)
    
    def readTokenAndLabelOfRawFile(self, lexicon, lexiconOfLabel, wordVecs, addWordUnkown, filters, indexesBySentence, labelsBySentence, token,lexiconFindInTrain):
        s = token.split('_')
        
        self.addToken(lexicon, wordVecs, addWordUnkown, filters, indexesBySentence, s[0],lexiconFindInTrain)
        self.addLabel(lexiconOfLabel, labelsBySentence, s[1])
    

    def readData(self, filename,lexicon,lexiconOfLabel, wordVecs=None, separateSentences= True, addWordUnkown=False,filters=[], lexiconFindInTrain=None):
        '''
        Read the data from a file and return a matrix which the first row is the words indexes  and second row is the labels values
        '''
        data = [[],[]]
        indexes = data[0]
        labels = data[1]
        
        func = self.readTokenAndLabelOfFileWithFeature if self.fileWithFeatures else self.readTokenAndLabelOfRawFile
        
        f = codecs.open(filename, 'r', 'utf-8')
        a = 0
        
        for line in f:
            
#             a +=1
#                   
#             if a ==10:
#                 break;
            
            line_split = line.split()
            # Ignore empty lines.
            if len(line_split) < 2:
                continue
            
            if separateSentences:
                indexesBySentence = []
                labelsBySentence = []
            else:
                indexesBySentence = indexes
                labelsBySentence = labels
            
            for token in line_split:
                func(lexicon, lexiconOfLabel, wordVecs, addWordUnkown, filters, indexesBySentence, labelsBySentence, token,lexiconFindInTrain)
            
            if len(indexesBySentence) != len(labelsBySentence):
                raise Exception('Número de tokens e labels não são iguais na linha: ' + line)
                         
            if separateSentences:
                indexes.append(indexesBySentence)
                labels.append(labelsBySentence)
                
        f.close()
        
#         list = []
#         i = 0
#         for l in data[0]:
#             if len(l) < 5:
#                 list.append((i,len(l)))
#             i+=1
# 
#         print list
        return data
    
    
    
 