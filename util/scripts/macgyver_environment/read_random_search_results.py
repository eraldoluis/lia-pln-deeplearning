import codecs
import json
import operator
import os.path
import sys
from ast import literal_eval

import numpy

"""
Reads and sort the results of the random search experiments.
Example of input:

{
    "input": "experiments/co_learning/co_learning_skip_context_0.txt",
    "pattern": "tests/ipython.out.%s",
    "datasets_info":[
            {
                "name": "weblogs",
                "number_tokens": 24025,
                "use_for_avg": true
            },
            {
                "name": "emails",
                "number_tokens": 29131,
                "use_for_avg": true
            },
            {
                "name": "wsj_test",
                "number_tokens": 32092,
                "use_for_avg": false
            }
    ]

}


"""


def main():
    # Get json with parameters of this script
    f = codecs.open(sys.argv[1], "r", "utf-8")
    js = json.load(f)

    # The file with the ouput of random_search.py
    inputFile = codecs.open(js["input"], "r", encoding="utf-8")

    # The pattern of files which contain the job outputs. Example: tests/ipython.out.%s
    patternFile = js["pattern"]

    """
    A list of objects. Ex:

    "datasets_info": [
        {
        "name": "weblogs", # Dataset name
        "number_tokens": 24025, # Number of tokens
        "use_for_avg": false # Use the result in this dataset to compute average.
        }

    ]
    """
    datasetsInfo = js["datasets_info"]

    datasetsCalculateAvg = []

    for dtInfo in datasetsInfo:
        if dtInfo["use_for_avg"]:
            datasetsCalculateAvg.append(dtInfo["number_tokens"])

    # Inverse dictionary. Dataset -> experiment -> acc
    experimentByDatasets = {"media": {}}

    for dtInfo in datasetsInfo:
        experimentByDatasets[dtInfo["number_tokens"]] = {}

    """
    We differentiate the experiments by their parameters.
    Each experiment has the following paremeters:
        "jobIds": job id of each round
        "jobAccs" The best accuracies of a job in each dataset. This is a dictionary
        "bestAccsAvg": Its store the values used to calculate accuracy average of a job.
        "accs": jobs accuracies for each dataset. This is a list.
        "avg": We calculate the average of each round. This average is called "experiment average". This parameter
            has the experiment average of each dataset.
        ""

    """
    #
    experiments = {}

    for line in inputFile:
        parametersStr, jobsIdsStr = line.split("} ", 1)
        parametersStr += "}"
        jobsIds = literal_eval(jobsIdsStr)

        experiments[parametersStr] = {
            "jobIds": [],
            "jobAccs": [],
            "bestAccsAvg": [],
        }

        for jobId in jobsIds:
            filePath = patternFile % jobId
            experiments[parametersStr]["jobIds"].append(jobId)

            if not os.path.isfile(filePath):
                print "The file %s doesn't exist" % filePath
                continue

            f = codecs.open(filePath, "r", encoding="utf-8")

            # The biggest accuracy of each dataset in a job. We differs a dataset from another by the token numbers.
            biggestAcc = {"media": -1}

            for dtInfo in datasetsInfo:
                biggestAcc[dtInfo["number_tokens"]] = -1

            experiments[parametersStr]["jobAccs"].append(biggestAcc)

            bestAccAvg = None
            accAvg = []

            # Reads accuracy of a job
            for l in f:
                try:
                    obj = json.loads(l)
                except:
                    continue

                msg = obj["message"]

                # Read messages with accuracy
                if "type" in msg and msg["type"] == "metric" and msg["name"] in ["AccTest", "AccDev"]:
                    numberTokens = msg["values"]["numExamples"]
                    acc = msg["values"]["accuracy"]

                    # Check if dataset result is to be read.
                    if numberTokens not in biggestAcc:
                        continue

                    if biggestAcc[numberTokens] < acc:
                        biggestAcc[numberTokens] = acc

                    if numberTokens in datasetsCalculateAvg:
                        accAvg.append(acc)

                    if len(accAvg) == len(datasetsCalculateAvg):
                        avg = numpy.asarray(accAvg).mean()

                        if avg > biggestAcc["media"]:
                            biggestAcc["media"] = avg
                            bestAccAvg = list(accAvg)

                        accAvg = []

            experiments[parametersStr]["bestAccsAvg"].append(bestAccAvg)

        # Transfer the found accuracy to experimentsAccs object
        experimentAccs = dict(experiments[parametersStr]["jobAccs"][0])

        for key in experimentAccs.keys():
            experimentAccs[key] = []

        for i, jobAcc in enumerate(experiments[parametersStr]["jobAccs"]):
            for key in experimentAccs.keys():
                print jobAcc
                experimentAccs[key].append(jobAcc[key])

        # Calculate the accuracy average of all runs
        experimentAvgs = dict(experimentAccs)

        for key in experimentAccs.keys():
            experimentAvg = numpy.asarray(experimentAccs[key]).mean()
            experimentAvgs[key] = experimentAvg
            experimentByDatasets[key][parametersStr] = experimentAvg

        experiments[parametersStr]["accs"] = experimentAccs
        experiments[parametersStr]["avg"] = experimentAvgs

    for numberTokens, accs in experimentByDatasets.iteritems():
        sorted_ = sorted(accs.items(), key=operator.itemgetter(1), reverse=True)
        datasetName = None

        for dtInfo in datasetsInfo:
            if dtInfo["number_tokens"] == numberTokens:
                datasetName = dtInfo["name"]

        print "%s\t%s" % (datasetName, numberTokens)

        print "30 Melhores"
        for i, o in enumerate(sorted_):
            if i > 30:
                break

            experimentParameter, accAvg = o
            experimentInfo = experiments[experimentParameter]

            print "%d\t%s\t%.4f\t%s\t%s" % (
            i + 1, experimentParameter, accAvg, experimentInfo["jobIds"], experimentInfo["accs"][numberTokens]),

            if numberTokens == "media":
                print "\t%s" % (experimentInfo["bestAccsAvg"]),
            print ""


if __name__ == '__main__':
    main()
