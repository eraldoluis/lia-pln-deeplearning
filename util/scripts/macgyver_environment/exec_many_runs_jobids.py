import codecs
import json
import os
import re
import sys
import time
import uuid
from ast import literal_eval

from experiments.domain_adaptation.random_search import executeSub


def readParametersFromLog(logPath):
    jsonParameters = {}

    for l in logPath:
        try:
            obj = json.loads(l)
        except:
            continue

        msg = obj["message"]

        if msg[:2] == "T(":
            jsonParameters = literal_eval("{" + re.sub(r'([a-zA-Z_]+)=', r'"\1":', msg[2:-1]) + "}")
            break

    return jsonParameters


def main():
    # The dir where a temporary json will be created
    dirPathJson = sys.argv[1]

    # Python script
    scriptPath = sys.argv[2]

    # The pattern of files which contain the job outputs. Example: tests/ipython.out.%s
    patternFile = sys.argv[3]

    # Jobs ids
    jobIds = literal_eval(sys.argv[4])

    # Parameters to print
    parametersToPrint = literal_eval(sys.argv[5])

    # Max memory usage in MB
    memory = int(sys.argv[6])

    begin = int(sys.argv[7])
    end = int(sys.argv[8])

    # Log file
    if len(sys.argv) > 9:
        log = codecs.open(sys.argv[9], "a", encoding="utf-8")
    else:
        log = None

    for jobId in jobIds:
        filePath = patternFile % jobId

        if not os.path.isfile(filePath):
            print "The file %s doesn't exist" % filePath
            continue

        f = codecs.open(filePath, "r", encoding="utf-8")

        jsonParameters = readParametersFromLog(f)

        parPrint = {}

        for parName in parametersToPrint:
            parPrint[parName] = jsonParameters[parName]

        newJobIds = []

        for runNumber in range(begin, end + 1):
            for key, value in jsonParameters.iteritems():
                if isinstance(value, (basestring, unicode)):
                    jsonParameters[key] = re.sub("run[0-9]+", "run%d" % runNumber, value)
                elif isinstance(value, (list)):
                    for idx, v in enumerate(jsonParameters[key]):
                        if isinstance(v, (basestring, unicode)):
                            jsonParameters[key][idx] = re.sub("run[0-9]+", "run%d" % runNumber, v)

            filePath = os.path.join(dirPathJson, uuid.uuid4().hex)
            with codecs.open(filePath, mode="w", encoding="utf-8") as f:
                json.dump(jsonParameters, f)

            print "Run %d" % runNumber

            out = executeSub(scriptPath, filePath, memory)

            r = re.findall("[0-9]+", out)

            if len(r) > 1:
                print "Tem mais de um numero na saida"
                sys.exit()

            newJobId = r[0]

            newJobIds.append(newJobId)

            print jsonParameters
            print "\n\n"
            time.sleep(1)

        newJobIds.insert(0,str(jobId))

        for j in newJobIds:
            print "%s\t" % (j),

        print ""

        if log:
            log.write("%s %s\n" %(parPrint,newJobIds))


if __name__ == '__main__':
    main()
