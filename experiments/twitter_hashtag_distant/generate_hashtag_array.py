# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import json


if __name__ == '__main__':
    dataset = '2.twitter_all-frequency-1kOrMore.txt'
    filepath = '/home/igor/Documents/LIA/Experimentos/twitter_sentiment_distant/' + dataset

    with open(filepath) as file:
        objs = json.load(file)

    print "["
    tags = np.asarray(objs["mostFreq"])[:50, :]
    for t in tags:
        print "\"" + t[2] + "\","
    print "]"


    #result = 0
    #for t in tags:
        #result += int(t)
    #print result
