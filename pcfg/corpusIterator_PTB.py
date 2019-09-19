# from https://www.asc.ohio-state.edu/demarneffe.1/LING5050/material/structured.html


from nltk.corpus import ptb
import os

def addTrees(sec, trees):
   secNum = ("" if sec >= 10 else "0") + str(sec)

   files = os.listdir("/u/scr/corpora/ldc/1999/LDC99T42/parsed/mrg/wsj/"+secNum)
   for name in files:
      for tree in ptb.parsed_sents("WSJ/"+secNum+"/"+name):
         trees.append(tree)


def getPTB(partition):
   trees = []
   if partition == "train":
     sections = range(0, 19)
   elif partition in ["dev", "valid"]: # 19-21
     sections = range(19, 22)
   elif partition == "test": # 22-24
     sections = range(22, 25)
   for sec in sections:
      print(sec)
      addTrees(sec, trees)
   return trees

#print(getPTB("train"))
import os
import random
import sys




class CorpusIterator_PTB():
   def __init__(self, language, partition="train"):
      data = getPTB(partition)
#      if shuffleData:
#       if shuffleDataSeed is None:
#         random.shuffle(data)
#       else:
#         random.Random(shuffleDataSeed).shuffle(data)

      self.data = data
      self.partition = partition
      self.language = language
      assert len(data) > 0, (language, partition)
   def permute(self):
      random.shuffle(self.data)
   def length(self):
      return len(self.data)
   def getSentence(self, index):
      result = self.data[index]
      return result
   def iterator(self):
     for sentence in self.data:
        yield sentence



