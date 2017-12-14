# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import json


if __name__ == '__main__':
    dataset = 'frequencia_1000xOrMore.json'
    filepath = '/home/igor/Desktop/LIA/top125/' + dataset

    with open(filepath) as file:
        objs = json.load(file)

    print "["
    tags = np.asarray(objs["mostFreq"])[:, 0]
    for t in tags:
        print "\"" + t + "\","
    print "]"

    #result = 0
    #for t in tags:
        #result += int(t)
    #print result
