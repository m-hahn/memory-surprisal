import sys
import random

path = "/u/scr/mhahn/CORPORA/ptb-ud2/ptb-ud2.conllu" # "/u/scr/corpora/ldc/2012/LDC2012T05/data/"+partition+".conll06"
with open(path, "r") as inFile:
    data = inFile.read().strip().split("\n\n")
    if len(data) == 1:
      data = data[0].split("\r\n\r\n")
    assert len(data) > 1
random.Random(0).shuffle(data)
assert len(data) > 0, (language, partition, files)


def readDependencyPTB(partition):
      if partition == "valid":
         partition = "dev"
      print >> sys.stderr, "Read "+str(len(data))+ " sentences from 1 "+partition+" datasets."

      if partition == "dev":
           return data[:5000]
      elif partition == "test":
           return data[5000:10000]
      else:
           return data[10000:]


