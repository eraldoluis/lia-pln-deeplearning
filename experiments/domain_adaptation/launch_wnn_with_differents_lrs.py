import codecs
import json
import os
import re
import sys
import time
import uuid

from experiments.domain_adaptation.random_search import executeSub


def main():
    # The dir where a temporary json will be created
    dirPathJson = sys.argv[1]

    # Python script
    scriptPath = sys.argv[2]

    # Parameters of the script
    jsonPath = sys.argv[3]

    # Max memory usage in MB
    memory = int(sys.argv[4])

    if len(sys.argv) > 5:
        # It's a json in a string format. This json has a object which
        # the names and values of its attributes are the name and values of parameters.
        # These parameters will overwrite the parameters values of the jsonPath
        commandParameters = json.loads(sys.argv[5])
    else:
        commandParameters = None

    if len(sys.argv) > 6:
        # Array of lrs in string format, for instance: "[0.1, 0.25, 0.5]"
        lrs = json.loads(sys.argv[6])
    else:
        lrs = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]

    with codecs.open(jsonPath, "r", encoding="utf-8") as f:
        jsonParameters = json.load(f)

    if commandParameters:
        for key,value in commandParameters.iteritems():
            jsonParameters[key] = value


    saveModel = jsonParameters["save_model"] if "save_model" in jsonParameters else None

    jobIds = []

    for lr in lrs:
        param = {}

        jsonParameters["lr"] = lr

        if "save_model" in jsonParameters and jsonParameters["save_model"]:
            jsonParameters["save_model"] = saveModel + "_" + str(lr)

        print "#######################################"
        print param,
        print "\t",
        i = 1

        filePath = os.path.join(dirPathJson, uuid.uuid4().hex)
        with codecs.open(filePath, mode="w", encoding="utf-8") as f:
            json.dump(jsonParameters, f)


        print "Run %d" % i

        out = executeSub(scriptPath, filePath, memory)

        r = re.findall("[0-9]+", out)

        if len(r) > 1:
            print "Tem mais de um numero na saida"
            sys.exit()

        jobId = r[0]

        jobIds.append(jobId)

        print jsonParameters
        print "\n\n"
        i += 1
        time.sleep(1)

    print commandParameters

    for jobId in jobIds:
        print "%s\t" %(jobId),


if __name__ == '__main__':
    main()