import argparse
import codecs
import os

import numpy as np
import matplotlib.pyplot as plt
import json


def createHistogram(epoch, average, variance, std, histogram, dirToSave):
    hist = {}
    intervals = set()

    # for interval, value in histogram.items():
    #     int1, int2 = interval.split(", ")
    #     int1 = float(int1[1:])
    #     int2 = float(int2[:-1])
    #
    #     hist[int1] = value
    #     intervals.add(int1)
    #     intervals.add(int2)

    intervals = np.round(sorted(histogram["bin"]),8)
    intervals[0] = -5
    print intervals
    values = histogram["values"]
    print values
    # values = []
    #
    # for interval, value in hist:
    #     values.append(float(value))

    values = np.asarray(values)
    values = values / float(values.sum())
    print values



    # print intervals
    #     print values

    # maxY = 0.

    a = []
    for idx, i in enumerate(intervals[:-1]):
        a.append("%g , %g" % (intervals[idx], intervals[idx + 1]))

    plt.bar(np.arange(len(intervals[:-1]))+ 30, values, width=1)
    # plt.ylim([0, maxY])
    plt.xticks(np.arange(len(intervals[:-1])) + 30.5, a, rotation=90)
    plt.title("Epoch %d" % (epoch))
    # plt.text(0.8, maxY, "Avg %.6f\nVariance %.6f\nSTD %0.6f " % (average, variance, std), fontsize=10)
    plt.savefig(os.path.join(dirToSave, "epoch_%d.png" % (epoch)))
    # plt.show()
    plt.close()


def processFile(logFile, metricName, saveFigPath):
    epoch = 0

    with codecs.open(logFile, "r", encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if len(l) == 0:
                continue

            o = json.loads(l)

            if "message" in o and "values" in o["message"]:
                name = o["message"]["name"]
                # epoch = o["message"]["epoch"]
                values = o["message"]["values"]
                if name == metricName:
                    epoch += 1
                    createHistogram(epoch, values["average"], values["variance"], values["std_deviation"],
                                    values["histogram"], saveFigPath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', help='the log file')
    parser.add_argument('metric_name', help='the metric name')
    parser.add_argument('images_dir_path', help='the directory where the images will be saved.')
    args = parser.parse_args()

    # processFile('"/home/irving/logs_jobs/log_wnn", "ActSupHidden", "/home/irving/logs_jobs/images/wnn")
    processFile(args.log, args.metric_name, args.images_dir_path)
