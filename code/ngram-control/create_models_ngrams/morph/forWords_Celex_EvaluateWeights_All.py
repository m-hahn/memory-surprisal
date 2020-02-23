# based on yWithMorphologySequentialStreamDropoutDev_Ngrams_Log.py

import random
import sys

objectiveName = "LM"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="Japanese_2.4")
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=0.0)
parser.add_argument("--gamma", dest="gamma", type=int, default=1)
parser.add_argument("--delta", dest="delta", type=float, default=1.0)
parser.add_argument("--cutoff", dest="cutoff", type=int, default=15)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)


assert args.alpha >= 0
assert args.alpha <= 1
assert args.delta >= 0
assert args.gamma >= 1





myID = args.idForProcess


TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams-morphology/"



posUni = set() 

posFine = set() 






from math import log, exp
from random import random, shuffle, randint, Random, choice

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_"]

from corpusIterator_V import CorpusIterator_V

originalDistanceWeights = {}

morphKeyValuePairs = set()

vocab_lemmas = {}

corpusTrain = CorpusIterator_V(args.language,"train", storeMorph=True).iterator(rejectShortSentences = False)
pairs = set()
counter = 0
data = []
for sentence in corpusTrain:
#    print(len(sentence))
    verb = []
    for line in sentence[::-1]:
#       print(line)
       if line["posUni"] == "PUNCT":
          continue
       verb.append(line)
       if line["posUni"] == "VERB":
          verb = verb[::-1]
#          print(verb)
#          print([x["dep"] for x in verb])
#          print([x["posUni"] for x in verb])
#          print([x["word"] for x in verb])
#          print([x["lemma"] for x in verb])
#          print([x["head"] for x in verb])
          for i in range(1,len(verb)):
            for j in range(1,i):
              pairs.add((verb[i]["lemma"], verb[j]["lemma"]))
              if (verb[j]["lemma"], verb[i]["lemma"]) in pairs:
                 print("======", (verb[i]["lemma"], verb[j]["lemma"]), [x["dep"] for x in verb], "".join([x["word"] for x in verb]))
          if len(verb) > 1:
            data.append([x["lemma"] for x in verb])
          counter += 1
          break
       if line["posUni"] not in ["AUX", "SCONJ"]:
          break
#       if line["dep"] not in ["aux"]:
 #         break
print(counter)
print(data)
print(len(data))

#quit()
import torch.nn as nn
import torch
from torch.autograd import Variable


import numpy.random



import torch.cuda
import torch.nn.functional


words = []


affixFrequencies = {}
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixFrequencies[affix] = affixFrequencies.get(affix, 0)+1

itos = set()
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    itos.add(affix)
itos = sorted(list(itos))
stoi = dict(list(zip(itos, range(len(itos)))))


print(itos)
print(stoi)

itos_ = itos[::]
shuffle(itos_)
weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))

weights = {'できる': 0, 'いける': 2, 'える': 4, 'でした': 6, 'たー': 8, 'める': 10, 'う': 12, 'ある': 14, 'られる': 16, 'ようだ': 18, 'まいる': 20, 'くださる': 22, '済み': 24, 'いらっしゃる': 26, 'おる': 28, 'かね ': 30, '始める': 32, '下さる': 34, '過ぎる': 36, 'ざるをえる': 38, 'あう': 40, 'ざるを得る': 42, 'こと': 44, 'ける': 46, 'てる': 48, '合う': 50, 'べる': 52, 'せる': 54, 'ので': 56, '込む': 58, 'から': 60, 'らしい': 62, '出来る': 64, '参る': 66, 'たい': 68, '頂く': 70, 'みたいだ': 72, 'なさる': 74, 'が': 76, 'がちだ': 78, 'そうだ': 80, 'ない': 82, 'づらい': 84, 'ちゃう': 86, 'ま~す': 88, 'ゆく': 90, 'の': 92, 'べし': 94, 'やる': 96, 'ん': 98, 'おく': 100, 'だ': 102, 'だめ': 104, '出す': 106, 'もらえる': 108, 'なければ': 110, 'いただく': 112, 'かもしれる': 114, 'なる': 116, 'させる': 118, '回る': 120, 'す ': 122, 'れる': 124, 'し': 126, 'くれる': 128, 'きる': 130, 'にくい': 132, 'ば': 134, 'がたい': 136, 'すぎる': 138, 'ます': 140, 'いく': 142, '易い': 144, '続ける': 146, 'みる': 148, 'まい': 150, 'ため': 152, 'やすい': 154, 'ね': 156, 'よい': 158, 'ほしい': 160, 'かける': 162, '直す': 164, 'らす': 166, 'た': 168, 'いる': 170, '行く': 172, 'しまう': 174, 'もらう': 176, '来る': 178, 'て': 180}

def getCorrectOrderCount(weights):
   correct = 0
   incorrect = 0
   for verb in data:
      for i in range(1, len(verb)):
         for j in range(1, i):
             weightI = weights[verb[i]]
             weightJ = weights[verb[j]]
             if weightI > weightJ:
               correct+=1
             else:
               incorrect+=1
   return correct/(correct+incorrect)

print(getCorrectOrderCount(weights))

