import re

class TransformNumberToZeroFilter:
    
    def filter(self,token):        
        return re.sub('[0-9]', '0', token)
        