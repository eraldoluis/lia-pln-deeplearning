from DataOperation import Lexicon
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
        
        for line in f:
            
            if re.search('[0-9] [0-9]', line):
                continue
            
            
            line_split = line.split(' ', 1 );
            
            lexicon.putWord(line_split[0]);
            wordVector.putWordVecStr(line_split[1])
            
        f.close()
        
        wordVector.setLenWordVectorAuto();
        
        return [lexicon,]
 