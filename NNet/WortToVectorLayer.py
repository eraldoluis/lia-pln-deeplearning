import theano
import theano.tensor as T

'''
    This class represent a window layer. 
'''
class WordToVectorLayer:
    
    def __init__(self, _input, Wv, wordSize, isToUpdateWordVector=True,updStrategy='normal',norm_coef=1.0):
        self.__wordSize = wordSize
        self.__Wv = Wv
        
        self.__output = T.flatten(self.__Wv[_input],2)
        self.__windowIdxs = _input
        self.__isToUpdateWordVector = isToUpdateWordVector
        self.__updStrategy = updStrategy
        self.__norm_coef = norm_coef 
        
                
    def getOutput(self):
        return self.__output
    
    def getUpdate(self,cost,learningRate):
        if not self.__isToUpdateWordVector:
            return None;
        
        gWordVector = -learningRate * T.grad(cost, self.__output);
        widowsIdxsFlatten = T.flatten(self.__windowIdxs, 1)
        gwordVectorFlatten = T.flatten(gWordVector, 1)
        reshapeSize = (gwordVectorFlatten.shape[0] / self.__wordSize, self.__wordSize)
        
        up = T.inc_subtensor(self.__Wv[widowsIdxsFlatten], T.reshape(gwordVectorFlatten, reshapeSize))
        
        if self.__updStrategy == 'normalize_mean':
            up = self.__norm_coef *(up - T.mean(up,axis=0))/T.ptp(up,axis=0)
            
        elif self.__updStrategy == 'zScore':
            up = self.__norm_coef * (up - T.mean(up,axis=0))/T.std(up,axis=0)
            
        
        return [(self.__Wv, 
                 up)]
        

