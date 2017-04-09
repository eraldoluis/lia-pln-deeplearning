import codecs
import json
import os
import re
import sys
import time
import uuid

from util.scripts.macgyver_environment.random_search import executeSub


def main():
    # The dir where a temporary json will be created
    dirPathJson = sys.argv[1]

    # Python script
    scriptPath = sys.argv[2]

    # Parameters of the script
    jsonPath = sys.argv[3]

    # Max memory usage in MB
    memory = int(sys.argv[4])


    begin = int(sys.argv[5])
    end = int(sys.argv[6])

    with codecs.open(jsonPath, "r", encoding="utf-8") as f:
        jsonParameters = json.load(f)

    jobIds = []
    i = 0

    for runNumber in range(begin,end+1):
        param = {}


        for key,value in jsonParameters.iteritems():
            if isinstance(value, (basestring, unicode)):
                jsonParameters[key] = re.sub("run[0-9]+","run%d"%(runNumber), value)
            elif isinstance(value, (list)):
                for idx, v  in enumerate(jsonParameters[key]):
                    if isinstance(v, (basestring, unicode)):
                        jsonParameters[key][idx] = re.sub("run[0-9]+","run%d"%(runNumber), v)


        filePath = os.path.join(dirPathJson, uuid.uuid4().hex)
        with codecs.open(filePath, mode="w", encoding="utf-8") as f:
            json.dump(jsonParameters, f)


        print "Run %d" % runNumber

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


    for jobId in jobIds:
        print "%s\t" %(jobId),


if __name__ == '__main__':
    main()
