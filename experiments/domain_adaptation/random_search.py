#!/usr/bin/env python
# -*- coding: utf-8 -*-


import codecs
import json
import os
import re
import scipy.stats
import subprocess
import sys
import time
import uuid

from numpy import random

"""
This script automatically launches experiments on german servers.
Each experiment has your parameters vary using random search and can have more than one round.

A example of a valid json:
{
	"jsons": ["/home/irving/workspace/nlp-deeplearning/parameters.txt"],
	"parameters" : [
		{
			"name": "lr",
			"type": "float",
			"round": 4,
			"a": 3,
			"b": 2
		},
		{
			"name": "window",
			"type": "list",
			"values": [3,5,7]
		},
		{
			"name": "hidden_size",
			"testXype": "int",
			"min": 25,
			"max": 200
		}
	],
	"nm": 5,
	"save_model_path": "test",
	"memory": 4000,
	"append_file": "/home/irving/aux/append",
	"scrip_path": "/home/irving/workspace/nlp-deeplearning/experiments/postag/wnn.py",
	"json_garbage": "/home/irving/aux/"
}
"""


def executeSub(scriptPath, filePath, memory, specificHost=None):
    input = """
    !/bin/sh
    #BSUB -J pln_theano_modifiacando_gereadord_do_tanh
    # use complete paths here
    #BSUB -e /home/irving/tests/ipython.err.%J
    #BSUB -o /home/irving/tests/ipython.out.%J

    echo "Doing my work now"

    # call your ipython script here
    cd /home/irving


    """

    input += "OMP_NUM_THREADS=1 THEANO_FLAGS='gcc.cxxflags=-L/opt/macgyver/usr/lib' " \
             "LD_LIBRARY_PATH=/opt/macgyver/usr/lib /opt/macgyver/scipy-stack-modern-openblas/bin/python -u " + scriptPath + " " + filePath

    arguments = ["bsub", "-n", "1", "-R", '"span[hosts=1] select[hname!=macgyver] rusage[mem=%d] "' % (memory)]

    if specificHost:
        arguments += ["-m", '"%s"'%(specificHost)]

    sp = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)

    out, err = sp.communicate(input=input.encode('utf-8'))

    if err == -1:
        raise Exception(err)

    if out:
        print "O job " + out + " foi criado"

    return out


def getNumbersOfJobs():
    sp = subprocess.Popen("bjobs", stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)

    out, err = sp.communicate()

    return len(re.findall("\n[0-9]+", out))


def main():
    # Get json with parameters of this script
    f = codecs.open(sys.argv[1], "r", "utf-8")
    js = json.load(f)

    # Execute many rounds of a same algorithm
    # For each round, you need to create a json with the basic parameters.
    # Each json will have different parameter values, however we will use the same values ​​for the parameters that vary.
    # Thus, the rounds can have different values for constant parameters and
    # will have the same values for not constant parameters.
    jsonsToRead = js["jsons"]

    # This is a list of objects.
    # Each object has parameter name and possible parameters values.
    parameters = js["parameters"]

    # The dir where a temporary json will be created
    dirPathJson = js["json_garbage"]

    # Number of jobs to be launched
    nm = js["nm"]

    # Python script
    scriptPath = js["scrip_path"]

    # The file which will be store the jobId and parameters of each job
    appendFile = js["append_file"]

    # Max memory used by script. This value is in MB.
    memory = js["memory"]

    if not isinstance(memory, int):
        raise Exception("Memory needs to be a integer")

    # Save model path
    if "save_model_path" in js:
        saveModelPath = js["save_model_path"]
    else:
        saveModelPath = None

    log = codecs.open(appendFile, "a", encoding="utf-8")

    baseJsons = []

    for j in jsonsToRead:
        with codecs.open(j, "r", encoding="utf-8") as f:
            baseJsons.append(json.load(f))

    # Create parameters that vary
    parametersValue = {}

    for param in parameters:
        name = param["name"]

        if param["type"] == "float":
            r = param["round"]
            a = param["a"]
            b = param["b"]
            parametersValue[name] = [round(10 ** (scipy.stats.uniform.rvs() * b - a), r) for _ in xrange(nm)]
        elif param["type"] == "int":
            min = param["min"]
            max = param["max"]
            parametersValue[name] = [random.randint(min, max) for _ in xrange(nm)]
        elif param["type"] == "list":
            parametersValue[name] = [random.choice(param["values"]) for _ in xrange(nm)]

    # Launch the 'nm' jobs
    for launchNum in xrange(nm):
        param = {}
        name = ""

        # Transfer the parameters values of a launch to a dictionary
        for paramName, paramValues in parametersValue.iteritems():
            name += paramName
            name += "_"
            name += str(paramValues[launchNum])
            name += "_"

            param[paramName] = paramValues[launchNum]

        print "#######################################"
        print param,
        print "\t",

        jobIds = []

        # Execute script python for each round
        for roundNum, jsonParameters in enumerate(baseJsons):

            # Change the parameter values that vary
            for k, v in param.iteritems():
                jsonParameters[k] = v

            if saveModelPath:
                saveModel = saveModelPath + "_" + name

                if len(baseJsons) > 1:
                    saveModel += "_%d" % (roundNum)

                jsonParameters["save_model"] = saveModel

            # Create a temporary json
            filePath = os.path.join(dirPathJson, uuid.uuid4().hex)
            with codecs.open(filePath, mode="w", encoding="utf-8") as f:
                json.dump(jsonParameters, f)

            # Launch job
            print "Run %d" % launchNum

            out = executeSub(scriptPath, filePath, memory)

            # Get job id
            r = re.findall("[0-9]+", out)

            if len(r) > 1:
                print "Tem mais de um numero na saida"
                sys.exit()

            jobIds.append(r[0])

            print jsonParameters
            print "\n\n"
            launchNum += 1
            time.sleep(1)

        st = str(param)
        st += " "
        st += str(jobIds)
        st += "\n"

        print st
        log.write(st)


if __name__ == '__main__':
    main()
