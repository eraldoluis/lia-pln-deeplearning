import os
import sys
import subprocess

def main():
    jsons = sys.argv[1:]

    for json in jsons:
        log_file = open(json + ".log", "w")
        proc = subprocess.Popen("python wnn_shortdoc_class.py " + json, stdout= log_file, shell=True)
        (out, err) = proc.communicate()
        log_file.close()

if __name__ == '__main__':
    main()