#!/usr/bin/env python
# -*- coding: utf-8 -*-

import crfsuite
from time import sleep
import os
import logging
import time
from util.util import unicodeToSrt
import itertools
import re
import codecs

# Inherit crfsuite.Trainer to implement message() function, which receives
# progress messages from a training process.
class Trainer(crfsuite.Trainer):
    def message(self, s):
        # Simply output the progress messages to STDOUT.
        logging.getLogger("Logger").info(s.rstrip())


class CRFSuite:
    
    __featuresTemplates = ((('w', 0),),
        (('w', -1),),
        (('w', 1),),
        (('w', -2),),
        (('w', 2),),
        )

    
    def __init__(self,unknownTokens,startSentenceSymbol,endSentenceSymbol,tokenLabelSeparator,filters):
        self.__logger = logging.getLogger("Logger")
        self.__unknownTokens = unknownTokens
        self.__startSentenceSymbol = startSentenceSymbol
        self.__endSentenceSymbol = endSentenceSymbol
        self.__tokenLabelSeparator = tokenLabelSeparator
        self.__filters = filters
        
    def __createFeature(self,label, value):
        return unicodeToSrt(label) + "=" + str(value)
        
    def __instances(self,fileRead, wordVectors, windowSize, useManualFeature):
        xseq = crfsuite.ItemSequence()
        yseq = crfsuite.StringList()
        defval = u''
        dataset = codecs.open(fileRead, 'r', 'utf-8')
        
        
        for line in dataset:
            i = 0
            tokens = []
            labels = []
            tokensWithLabels = line.rstrip().split(' ')
            
    #         currentTime = calendar.timegm(time.gmtime())
            
    #         if currentTime - instances.lastTimePrintedMsg > 30.0:
    #             instances.lastTimePrintedMsg = currentTime
    #             logger.info("Processing File. Memory usage: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    
            for token in tokensWithLabels:
                if token.isspace() or not token:
                    continue
                
                t = token.rsplit(self.__tokenLabelSeparator,1)
                
                if len(t[1]) == 0:
                    logging.getLogger("Logger").warn("It was not found the label from "\
                                 "the token " + token + ". We give to this token "\
                                 " a label equal to"\
                                 " the tokenLabelSeparator( " + self.__tokenLabelSeparator +")" )
                      
                    t[1] = self.__tokenLabelSeparator
                
                try:
                    tokens.append(t[0])
                    labels.append(t[1])
                except Exception:
                    print t
                    print line
                    
                    
            
            halfWindowSize = windowSize / 2
            
            for i in range(len(tokens)):
                beginIndex = i - halfWindowSize
                item = crfsuite.Item()
                
                
                if useManualFeature:
                    item.append(crfsuite.Attribute(self.__createFeature("num" , str(int(re.search('\d', tokens[i]) is not None)))))
                    item.append(crfsuite.Attribute(self.__createFeature("cap" , str(any(c.isupper() for c in tokens[i])))))
                    item.append(crfsuite.Attribute(self.__createFeature("hyp" , str(int(re.search('-', tokens[i]) is not None)))))
                 
                    # prefixos
                    item.append(crfsuite.Attribute(self.__createFeature("p1", tokens[i][0] if len(tokens[i]) >= 1 else defval)))
                    item.append(crfsuite.Attribute(self.__createFeature("p2", tokens[i][:2] if len(tokens[i]) >= 2 else defval)))
                    item.append(crfsuite.Attribute(self.__createFeature("p3", tokens[i][:3] if len(tokens[i]) >= 3 else defval)))
                    item.append(crfsuite.Attribute(self.__createFeature("p4", tokens[i][:4] if len(tokens[i]) >= 4 else defval)))
                     
                    # sufixos
                    item.append(crfsuite.Attribute(self.__createFeature("s1", tokens[i][-1] if len(tokens[i]) >= 1 else defval)))
                    item.append(crfsuite.Attribute(self.__createFeature("s2", tokens[i][-2:] if len(tokens[i]) >= 2 else defval)))
                    item.append(crfsuite.Attribute(self.__createFeature("s3", tokens[i][-3:] if len(tokens[i]) >= 3 else defval)))
                    item.append(crfsuite.Attribute(self.__createFeature("s4", tokens[i][-4:] if len(tokens[i]) >= 4 else defval)))
                     
                     
                    for featureTemplate, indexFeature in itertools.izip(CRFSuite.__featuresTemplates, range(len(CRFSuite.__featuresTemplates))):
                        namesFeature = []
                        valuesFeature = []
                         
                        for name, index in featureTemplate:
                            namesFeature.append(name + "[" + str(index) + "]")
                            valuesFeature.append(tokens[i + index] if i + index >= 0 and i + index < len(tokens)  else defval)
                         
                        names = "|".join(namesFeature)
                        values = "|".join(valuesFeature)
                         
                        item.append(crfsuite.Attribute(self.__createFeature(names, values)))
        
                for j in range(windowSize):            
                    index = beginIndex + j
                    label = str(j) + u'|'
                    
                    if index < 0:
                        token = self.__startSentenceSymbol
                    elif index >= len(tokens):
                        token = self.__endSentenceSymbol
                    else:
                        token = tokens[index]
                        
                        for filter in self.__filters:
                            token = filter.filter(token)
                    
                    k = 0
                    for wordvector in wordVectors:
                        if token in wordvector:
                            wv = wordvector[token]
                        else:
                            for unknownToken in self.__unknownTokens:
                                if unknownToken in wordvector:
                                    wv = wordvector[unknownToken]
                                    break;
                        
                        for number in wv:
                            item.append(crfsuite.Attribute(self.__createFeature(label + str(k),'_'), number))
                            
                            k += 1
                            
                xseq.append(item)
                yseq.append(unicodeToSrt(labels[i]))
                
            yield xseq, tuple(yseq)
            xseq = crfsuite.ItemSequence()
            yseq = crfsuite.StringList()
        
    def __printFeatures(self,xseq, yseq):
        line = ''  
        self.__logger.info("Imprimendo features the instancia")
        
        for x, y in itertools.izip(xseq,yseq):
            line = y;
            
            for attribute in x:
                line += ' ' + attribute.attr + ":" + str(attribute.value)
    
        
            self.__logger.info(line)
    
        self.__logger.info("")
    
    def train(self,source,dev,wordVectors,windowSizeFeatures,useManualFeature,
              numberEpoch,noTestByEpoch,modelPath,unknownTokens,nmFeatureSentenceToPrint = 2):
        self.__logger.info("Gerando featuresdo treino")
        trainer = Trainer()
        
        for xseq, yseq in self.__instances(source, wordVectors, windowSizeFeatures, useManualFeature):
            trainer.append(xseq, yseq, 0)
            if nmFeatureSentenceToPrint > 0:
                self.__printFeatures(xseq, yseq)
                nmFeatureSentenceToPrint -=1
                
        self.__logger.info("Terminando de gerar features do treino")
        
        
        trainer.select('lbfgs', 'crf1d')
        
        if numberEpoch > 0:
            trainer.set('max_iterations', str(numberEpoch))
        
        for name in trainer.params():
            self.__logger.info(name + " " + trainer.get(name) + " " + trainer.help(name))
         
        
        sleep(5)
    
        if not noTestByEpoch:
            holdout = 2
            
            self.__logger.info( "Gerando features do teste")
            for xseq, yseq in self.__instances(dev, wordVectors, windowSizeFeatures, useManualFeature):
                trainer.append(xseq, yseq, 1)
                
                if nmFeatureSentenceToPrint > 0:
                    self.__printFeatures(xseq, yseq)
                    nmFeatureSentenceToPrint-=1
                    
            self.__logger.info( "Terminando de gerar features do teste")
        else:
            holdout = -1
        
          
        self.__logger.info( "Comecando Treino: " + time.ctime())
        trainer.train(modelPath, holdout)
        self.__logger.info( "Terminando Treino: " + time.ctime())
    
    
    def test(self,target,modelPath,wordVectors,windowSizeFeatures,useManualFeature,numberEpoch,
             noTestByEpoch,unknownTokens,nmFeatureSentenceToPrint = 2):
        
        sleep(5)
    
        tagger = crfsuite.Tagger()
        tagger.open(modelPath)
        
        total = 0
        numberCorrect = 0;
        
        self.__logger.info( "Comecando teste: " + time.ctime())
        for xseq, yseqcor in self.__instances(target, wordVectors, windowSizeFeatures, useManualFeature):
            # Tag the sequence.
            if nmFeatureSentenceToPrint > 0:
                self.__printFeatures(xseq, yseqcor)
                nmFeatureSentenceToPrint-=1
    
            tagger.set(xseq)
            
            
            # Obtain the label sequence predicted by the tagger.
            yseqpred = tagger.viterbi()
            
            for cor, pred in  itertools.izip(yseqcor, yseqpred):
    #             logger.info("cor | pred : " + cor + " | " + pred);
                if cor == pred:
                    numberCorrect += 1
                total += 1
        
        
        return (numberCorrect,total)
    
