class LogPrior:

  def __init__(self, sigma):
    self.sigma = sigma;
  
  def compute(x, grad):
    val = 0.0;

    for (i in range(0,len(x))):
      val += (x[i] *x[i]) / (2.0 * self.sigma*self.sigma);
      grad[i] += x[i] / (sigma*sigma);
    return val
  

