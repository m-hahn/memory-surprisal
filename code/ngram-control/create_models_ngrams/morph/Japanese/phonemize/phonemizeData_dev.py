# based on yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py

import random
import sys
import romkan

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="Japanese-GSD_2.4")
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=0.0)
parser.add_argument("--gamma", dest="gamma", type=int, default=1)
parser.add_argument("--delta", dest="delta", type=float, default=1.0)
parser.add_argument("--cutoff", dest="cutoff", type=int, default=3)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)


assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1





myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"



posUni = set() 

posFine = set() 






from math import log, exp
from random import random, shuffle, randint, Random, choice

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator_V import CorpusIterator_V

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}


def processVerb(verb):
    if len(verb) > 0:
      if "VERB" in [x["posUni"] for x in verb[1:]]:
        print([x["word"] for x in verb])
      data.append(verb)

corpusTrain = CorpusIterator_V(args.language,"dev", storeMorph=True).iterator(rejectShortSentences = False)
pairs = set()
counter = 0
data = []
for sentence in corpusTrain:
#    print(len(sentence))
    verb = []
    for line in sentence:
       if line["posUni"] == "PUNCT":
          processVerb(verb)
          verb = []
          continue
       elif line["posUni"] == "VERB":
          processVerb(verb)
          verb = []
          verb.append(line)
       elif line["posUni"] == "AUX" and len(verb) > 0:
          verb.append(line)
       elif line["posUni"] == "SCONJ" and line["word"] == 'て':
          verb.append(line)
          processVerb(verb)
          verb = []
       else:
          processVerb(verb)
          verb = []
print(len(data))
#quit()
print(counter)
#print(data)
print(len(data))

#quit()
import torch.nn as nn
import torch
from torch.autograd import Variable


import numpy.random



import torch.cuda
import torch.nn.functional


words = []

affixFrequency = {}
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixLemma = affix["lemma"]
    affixFrequency[affixLemma] = affixFrequency.get(affixLemma, 0)+1


itos = set()
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixLemma = affix["lemma"]
    itos.add(affixLemma)
itos = sorted(list(itos))
stoi = dict(list(zip(itos, range(len(itos)))))


print(itos)
print(stoi)

itos_ = itos[::]
shuffle(itos_)
weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))

raw2Tagged = dict()

with open("../data/extractedVerbs_dev.txt", "r") as inFileRaw:
  with open("../data/extractedVerbs_tagged_dev.txt", "r") as inFileTagged:
    try:
     for index in range(1000000):
       raw = next(inFileRaw).strip()
       tagged = next(inFileTagged).strip()
       raw2Tagged[raw] = tagged
    except StopIteration:
       _ = 0

raw2Phonemized = {}
for line in data:
 # print(" ".join([x["word"] for x in line]))
  raw = "".join([x["word"] for x in line])
  tagged = [x.split("/") for x in raw2Tagged[raw].split(" ")]
  if raw == "".join([x[2] for x in tagged]):
     for x in line:
        x["hiragana"] = x["word"]

     raw_ = " ".join([x["word"] for x in line])
     raw2Phonemized[raw_] = " ".join([x["hiragana"] for x in line])


     continue
  iTagged = 0
  jTagged = 0
  iLine = 0
  jLine = 0
  line[iLine]["hiragana"] = ""

  print(raw)
  for i in range(len(raw)):
    print(iTagged, jTagged, tagged, raw)    
    jTagged += 1
    if jTagged == len(tagged[iTagged][0]):
      line[iLine]["hiragana"] += tagged[iTagged][2]
      iTagged += 1
      jTagged = 0 
    jLine += 1
    if jLine == len(line[iLine]["word"]):
      iLine += 1
      if iLine < len(line):
        line[iLine]["hiragana"] = ""
      jLine = 0 
     
#  print(line)
  assert "".join([x["hiragana"] for x in line]) == "".join([x[2] for x in tagged])
#  print(line)
#  print(tagged)
#  print([x["word"] for x in line])
#  print("RAW", raw)
#  print(max([ord(y) for y in raw]))
  hasFoundProblem = []
  for index, x in enumerate(line):
#     print(x["word"], "==", x["hiragana"])
     if x["word"] != x["hiragana"] and max([ord(y) for y in x["word"]]) < 200 and x["word"] not in ["fax", "pr"]:
        hasFoundProblem.append(index)
     elif x["hiragana"] == "":
        hasFoundProblem.append(index)
#     assert x["hiragana"]  != "", tagged
  if len(hasFoundProblem)>0:
    assert len(line) > 1
 #   print(hasFoundProblem)
    if hasFoundProblem == [0]:
      assert len(line) > 1
      if line[1]["hiragana"].endswith(line[1]["word"]): 
         line[0]["hiragana"] = line[1]["hiragana"][:-len(line[1]["word"])]
         line[1]["hiragana"] = line[1]["word"]
      elif len(line[1]["hiragana"]) == len(line[0]["word"]) + len(line[1]["word"]):
         line[0]["hiragana"] = line[1]["hiragana"][:len(line[0]["word"])]
         line[1]["hiragana"] = line[1]["hiragana"][len(line[0]["word"]):]
      elif line[0]["word"] == "出":
         assert line[1]["hiragana"].startswith("で")
         line[0]["hiragana"] = "で"
         line[1]["hiragana"] = line[1]["hiragana"][1:]
      elif line[0]["word"] == "見":
         assert line[1]["hiragana"].startswith("み")
         line[0]["hiragana"] = "み"
         line[1]["hiragana"] = line[1]["hiragana"][1:]
      elif line[1]["hiragana"].count(line[0]["word"][-1]) == 1:
         line[0]["hiragana"] = line[1]["hiragana"][:line[1]["hiragana"].index(line[0]["word"][-1])]
         line[1]["hiragana"] = line[1]["hiragana"][line[1]["hiragana"].index(line[0]["word"][-1]):]
      else:
         assert False
    elif hasFoundProblem == [1]:
      if line[1]["word"] + line[2]["word"] == line[2]["hiragana"]:
        line[1]["hiragana"] = line[1]["word"]
        line[2]["hiragana"] = line[2]["word"]
      else:
        assert False
    elif hasFoundProblem == [2]:
      if line[2]["word"] + line[3]["word"] == line[3]["hiragana"]:
        line[2]["hiragana"] = line[2]["word"]
        line[3]["hiragana"] = line[3]["word"]
      else:
        assert False
    elif hasFoundProblem == [0,1]:
       if line[0]["word"] == "見":
         line[0]["hiragana"] = "み"
       if line[0]["word"] == "つ":
          line[0]["hiragana"] = line[0]["word"]
       if line[1]["word"] == "込":
         line[1]["hiragana"] = "こ"
       if line[1]["word"] == "け":
          line[1]["hiragana"] = line[1]["word"]
       line[2]["hiragana"] = line[2]["hiragana"][len(line[0]["hiragana"])+len(line[1]["hiragana"]):]
#  print(line)
#  assert "".join([x["hiragana"] for x in line]) == ("".join([x[2] for x in tagged])).replace("ＰＲ", "pr")
  for index, x in enumerate(line):
 #    print(x["word"], "==", x["hiragana"])
     if x["word"] != x["hiragana"] and max([ord(y) for y in x["word"]]) < 200 and x["word"] not in ["pr", "fax"]:
        assert False
     elif x["hiragana"] == "":
        assert False
  raw_ = " ".join([x["word"] for x in line])
  raw2Phonemized[raw_] = " ".join([x["hiragana"] for x in line])

with open("../data/extractedVerbs_hiragana_dev.txt", "w") as outFileTagged:
  for raw_ in raw2Phonemized:
     print(raw_, "\t", raw2Phonemized[raw_], file=outFileTagged)

