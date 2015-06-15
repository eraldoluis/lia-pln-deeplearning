#import theano
import theano.tensor as T

'''
    This class represent a window layer. 
'''
class WordToVectorLayer:
    
    def __init__(self, _input, Wv, wordSize, isToUpdateWordVector=True):
        self.__wordSize = wordSize
        self.__Wv = Wv
        self.__output = T.flatten(self.__Wv[_input],2)
        self.__windowIdxs = _input
        self.__isToUpdateWordVector = isToUpdateWordVector
        
                
    def getOutput(self):
        return self.__output
    
    def getUpdate(self,cost,learningRate):
        if not self.__isToUpdateWordVector:
            return None;
        
        gWordVector = -learningRate * T.grad(cost, self.__output);
        widowsIdxsFlatten = T.flatten(self.__windowIdxs, 1)
        gwordVectorFlatten = T.flatten(gWordVector, 1)
        reshapeSize = (gwordVectorFlatten.shape[0] / self.__wordSize, self.__wordSize)
        
        return [(self.__Wv,
                    T.inc_subtensor(self.__Wv[widowsIdxsFlatten], T.reshape(gwordVectorFlatten, reshapeSize)))]
