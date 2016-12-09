import argparse
import codecs
import os

import numpy as np
import matplotlib.pyplot as plt
import json


def createLinePlot(dict,saveFigPath):


    for metricName, d in dict.items():
        plt.figure()
        x = range(len(d["avg"]))

        plt.errorbar(x, d["avg"], yerr=d["std"])

    plt.show()



def processFile(logFile, metricNames, saveFigPath):
    with codecs.open(logFile, "r", encoding="utf-8") as f:
        dict = {}

        for mn in metricNames:
            dict[mn] = {
                "avg": [],
                "std": [],
            }

        for l in f:
            l = l.strip()
            if len(l) == 0:
                continue

            o = json.loads(l)
            if "message" in o and "values" in o["message"]:
                name = o["message"]["name"]
                epoch = o["message"]["epoch"]
                values = o["message"]["values"]
                if name in metricNames:
                    dict[name]["avg"].append(values["average"])
                    dict[name]["std"].append(values["std_deviation"])

        createLinePlot(dict,saveFigPath)

        print dict



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', help='the log file')
    parser.add_argument('images_dir_path', help='the directory where the images will be saved.')
    parser.add_argument('metric_names', help='the metric names', nargs="+")
    args = parser.parse_args()

    # processFile('"/home/irving/logs_jobs/log_wnn", "ActSupHidden", "/home/irving/logs_jobs/images/wnn")
    processFile(args.log, args.metric_names, args.images_dir_path)
