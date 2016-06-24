class Scorer:
  def score(self,data):
    self.trueEntities = []
    self.guessEntities = []
    prevLabel = "O";
    start = 0;

    for i in range(0,len(data)):
      label = data[i].label;

      if (label==("PERSON") and prevLabel==("O")):
        start = i;        
      elif (label==("O") and prevLabel==("PERSON")):
        self.trueEntities.append([start,i]);
      prevLabel = label;
    

    prevLabel = "O";
    
    for i in range(0,len(data)):
      label = data[i].guessLabel;
      if (label==("PERSON") and prevLabel==("O")): 
	start = i;        
      elif (label==("O") and prevLabel==("PERSON")):
	self.guessEntities.append([start,i]);
      prevLabel = label;
    

    s = [filter(lambda x: x in self.trueEntities, sublist) for sublist in self.guessEntities]

    tp = float(len(s));

    prec = tp / float(len(self.guessEntities));
    recall = tp / float(len(self.trueEntities));
    f = (2.0 * prec * recall) / (prec + recall);

    print "precision = ", prec
    print "recall = ", recall
    print "F1 = ",f
    #print "F1 = ",f
  

class Pair:
  def __init__(self,first, second):
    self.first = first;
    self.second = second;
  
  def hashCode (self):
    return (self.first << 16) ** second;
  
  def equals(self,o):
    if (isinstance(o, Pair)==False):
      return False;
    return (self.first == o.first and self.second == o.second);
  
  def toString(self):
    return "(",self.first, ", ",self.second,")";
  
  

    
  