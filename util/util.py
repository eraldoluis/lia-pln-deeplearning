import subprocess
import os


def execProcess(cmd, logger, working_director=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE,cwd=working_director)
    
    while p.poll() is None:
        l = p.stdout.readline()  # This blocks until it receives a newline.
        print l.rstrip()
    
    print p.stdout.read().rstrip()
    
def unicodeToSrt(s):
    return str(s.encode('utf-8'))

def getFileNameInPath():
    return os.path.split(str)[1]

def removeExtension(file):    
    return os.path.splitext(file)[0]