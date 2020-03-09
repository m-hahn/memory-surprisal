import gzip
from random import shuffle, choice
import random
from math import log, exp

TARGET_DIR = "/u/scr/mhahn/deps/memory-need-ngrams-morphology-optimized/"


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="Japanese_2.4")
parser.add_argument("--model", dest="model", type=str)
parser.add_argument("--alpha", dest="alpha", type=float, default=0.0)
parser.add_argument("--gamma", dest="gamma", type=int, default=1)
parser.add_argument("--delta", dest="delta", type=float, default=1.0)
parser.add_argument("--cutoff", dest="cutoff", type=int, default=3)
parser.add_argument("--idForProcess", dest="idForProcess", type=int, default=random.randint(0,10000000))
import random



args=parser.parse_args()
print(args)



myID = random.randint(10000,1000000000)

formsCounter = 0


MAX_FORMS = 5000 #1000000

forms_data = []

def finishForm(head, suffixes):
  global formsCounter
  formsCounter += 1
  if formsCounter % 100 == 0:
     print(formsCounter)
  forms_data.append((head, suffixes))
    
affixMorphemes = set(['き/語尾/き', '始め/動詞/はじめ', 'た/助動詞/た', 'る/語尾/る', 'な/動詞/な', 'っ/語尾/っ', 'でき/動詞/でき', 'な/助動詞/な', 'く/語尾/く', 'な/動詞/な', 'る/語尾/る', 'さ/動詞/さ', 'れ/助動詞/れ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'み/語尾/み', '始め/動詞/はじめ', 'た/助動詞/た', 'が/語尾/が', 'れ/助動詞/れ', 'た/助動詞/た', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'だ/助動詞/だ', 'ろ/語尾/ろ', 'う/助動詞/う', 'わ/語尾/わ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'ば/語尾/ば', 'な/助動詞/な', 'い/語尾/い', 'ら/語尾/ら', 'せ/助動詞/せ', 'た/助動詞/た', 'られ/助動詞/られ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'られ/助動詞/られ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'く/語尾/く', 'だ/助動詞/だ', 'ろ/語尾/ろ', 'う/助動詞/う', 'かっ/語尾/かっ','り/語尾/り', 'ま/助動詞/ま', 'し/語尾/し', 'た/助動詞/た','に/助動詞/に', 'な/動詞/な', 'ら/語尾/ら', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'た/助動詞/た', 'らし/助動詞/らし', 'い/語尾/い', 'せ/語尾/せ', 'られ/助動詞/られ', 'た/助動詞/た', 'と/助動詞/と', 'な/動詞/な', 'っ/語尾/っ', 'た/助動詞/た', 'う/語尾/う', 'だ/助動詞/だ', 'ろ/語尾/ろ', 'う/助動詞/う', 'い/語尾/い', 'らし/助動詞/らし', 'い/語尾/い', 'させ/助動詞/させ', 'はじめ/動詞/はじめ', 'し/動詞/し', 'う/助動詞/う', 'る/語尾/る', 'た/語尾/た', 'な/助動詞/な', 'い/語尾/い', 'さ/語尾/さ', 'な/助動詞/な', 'い/語尾/い', 'ま/語尾/ま', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'り/語尾/り', 'おろ/動詞/おろ', 'し/語尾/し', 'させ/助動詞/させ', 'か/語尾/か', 'な/助動詞/な', 'い/語尾/い', 'ら/語尾/ら', 'ず/助動詞/ず', 'し/動詞/し', 'な/助動詞/な', 'く/語尾/く', 'な/動詞/な', 'っ/語尾/っ', 'た/助動詞/た', 'させ/助動詞/させ', 'た/語尾/た', 'れ/助動詞/れ', 'た/助動詞/た', 'き/語尾/き', 'まく/動詞/まく', 'り/語尾/り', 'はじめ/動詞/はじめ', 'た/助動詞/た', 'て/助動詞/て', 'られ/助動詞/られ', 'よう/助動詞/よう', 'く/語尾/く', 'あ/動詞/あ', 'る/語尾/る', 'でき/動詞/でき', 'ま/助動詞/ま', 'す/語尾/す', 'せ/動詞/せ', 'られ/助動詞/られ', 'る/語尾/る', 'り/語尾/り', 'ま/助動詞/ま', 'せ/語尾/せ', 'ん/助動詞/ん', 'ら/語尾/ら', 'れ/助動詞/れ', 'た/助動詞/た', 'く/語尾/く', 'あ/動詞/あ', 'り/語尾/り', 'ま/助動詞/ま', 'せ/語尾/せ', 'ん/助動詞/ん', 'つ/動詞/つ', 'く/語尾/く', 'さ/語尾/さ', 'れ/助動詞/れ', 'た/助動詞/た', 'い/語尾/い', 'ま/助動詞/ま', 'せ/語尾/せ', 'ん/助動詞/ん', 'られ/助動詞/られ', 'ま/助動詞/ま', 'せ/語尾/せ', 'ん/助動詞/ん', 'く/語尾/く', 'で/助動詞/で', 'あ/動詞/あ', 'る/語尾/る', 'し/動詞/し', 'ま/助動詞/ま', 'せ/語尾/せ', 'ん/助動詞/ん', 'ら/語尾/ら', 'ぬ/助動詞/ぬ', 'ぜ/語尾/ぜ', 'られ/助動詞/られ', 'た/助動詞/た', 'い/語尾/い', 'ま/助動詞/ま', 'せ/語尾/せ', 'ん/助動詞/ん', 'ぜ/語尾/ぜ', 'られ/助動詞/られ', 'た/助動詞/た', 'つづけ/動詞/つづけ', 'す/語尾/す', 'べ/助動詞/べ', 'き/語尾/き', 'だ/助動詞/だ', 'じ/語尾/じ', 'させ/助動詞/させ', 'り/語尾/り', '続け/動詞/つづけ', 'かっ/語尾/かっ', 'た/助動詞/た', 'かっ/語尾/かっ'])

def processVerb(verb):
   head = verb[0]
   suffixes = tuple(verb[1:])
   if suffixes == ('る/語尾/る',):
      finishForm(head, suffixes)
   elif suffixes == ('た/助動詞/た',):
      finishForm(head, suffixes)
   elif suffixes == ('う/語尾/う',):
      finishForm(head, suffixes)
   elif suffixes == ('い/語尾/い',):
      finishForm(head, suffixes)
   elif suffixes == ('し/動詞/し', 'た/助動詞/た') or suffixes == ('する/動詞/する',) or suffixes == ('っ/語尾/っ', 'た/助動詞/た') or suffixes == ('む/語尾/む',) or suffixes == ('あ/動詞/あ', 'る/語尾/る') or suffixes == ('あ/動詞/あ', 'っ/語尾/っ', 'た/助動詞/た') or suffixes == ('さ/動詞/さ', 'れ/助動詞/れ', 'た/助動詞/た') or suffixes == ('られ/助動詞/られ', 'る/語尾/る') or suffixes == ('さ/語尾/さ', 'れ/助動詞/れ', 'た/助動詞/た') or suffixes == ('わ/語尾/わ', 'な/助動詞/な', 'い/語尾/い') or suffixes == ('でき/動詞/でき', 'る/語尾/る') or suffixes == ('で/助動詞/で', 'あ/動詞/あ', 'っ/語尾/っ', 'た/助動詞/た') or suffixes == ('か/語尾/か', 'れ/助動詞/れ', 'た/助動詞/た') or suffixes == ('だっ/助動詞/だっ', 'た/助動詞/た') or suffixes == ('く/語尾/く',) or suffixes == ('し/語尾/し', 'た/助動詞/た') or suffixes ==  ('な/助動詞/な', 'い/語尾/い') or suffixes == ('で/助動詞/で', 'あ/動詞/あ', 'る/語尾/る') or suffixes == ('さ/動詞/さ', 'れ/助動詞/れ', 'る/語尾/る') or len(suffixes) == 0 or suffixes == ('い/語尾/い', 'た/助動詞/た') or suffixes == ('れ/助動詞/れ', 'る/語尾/る') or suffixes == ('す/語尾/す',) or suffixes == ('られ/助動詞/られ', 'た/助動詞/た') or suffixes == ('ら/語尾/ら', 'な/助動詞/な', 'い/語尾/い') or suffixes == ('ん/語尾/ん', 'だ/助動詞/だ') or suffixes == ('じ/語尾/じ', 'た/助動詞/た') or suffixes == ('つ/語尾/つ',) or suffixes == ('ら/語尾/ら', 'れ/助動詞/れ', 'た/助動詞/た') or suffixes == ('す/語尾/す', 'な/助動詞/な', 'り/語尾/り') or suffixes == ('ば/語尾/ば', 'れ/助動詞/れ', 'た/助動詞/た') or suffixes == ('さ/動詞/さ', 'せ/助動詞/せ', 'た/助動詞/た') or suffixes == ('な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た') or suffixes == ('ら/語尾/ら', 'れ/助動詞/れ', 'る/語尾/る') or suffixes == ('ら/語尾/ら', 'れ/助動詞/れ', 'る/語尾/る') or suffixes == ('さ/語尾/さ', 'れ/助動詞/れ', 'る/語尾/る') or suffixes == ('か/語尾/か', 'せ/助動詞/せ', 'た/助動詞/た') or suffixes == ('わ/語尾/わ', 'れ/助動詞/れ', 'た/助動詞/た') or suffixes == ('さ/動詞/さ', 'れ/助動詞/れ', '始め/動詞/はじめ', 'た/助動詞/た') or suffixes == ('ま/語尾/ま', 'れ/助動詞/れ', 'る/語尾/る') or suffixes == ('ま/語尾/ま', 'れ/助動詞/れ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('わ/語尾/わ', 'れ/助動詞/れ', 'る/語尾/る') or suffixes == ('ば/語尾/ば', 'れ/助動詞/れ', 'る/語尾/る') or suffixes == ('し/動詞/し', '続け/動詞/つづけ', 'た/助動詞/た') or suffixes == ('し/動詞/し', 'な/助動詞/な', 'い/語尾/い') or suffixes == ('じ/語尾/じ', 'られ/助動詞/られ', 'た/助動詞/た') or suffixes == ('ぶ/語尾/ぶ',) or suffixes == ('じ/語尾/じ', 'られ/助動詞/られ', 'た/助動詞/た') or suffixes == ('じ/語尾/じ', 'られ/助動詞/られ', 'る/語尾/る') :
      finishForm(head, suffixes)
   elif suffixes == ('でき/動詞/でき', 'な/助動詞/な', 'い/語尾/い') or suffixes == ('する/語尾/する',) or suffixes == ('始め/動詞/はじめ', 'る/語尾/る') or suffixes ==  ('ぜ/語尾/ぜ', 'られ/助動詞/られ', 'る/語尾/る') or suffixes == ('たが/助動詞/たが', 'い/動詞/い', 'た/助動詞/た') or suffixes == ('ふ/語尾/ふ',) or suffixes == ('し/動詞/し', '始め/動詞/はじめ', 'た/助動詞/た') or suffixes == ('さ/動詞/さ', 'せ/助動詞/せ', 'る/語尾/る') or suffixes == ('さ/語尾/さ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た') or suffixes == ('わ/語尾/わ', 'せ/助動詞/せ', 'た/助動詞/た') or suffixes== ('しめ/助動詞/しめ', 'た/助動詞/た') or suffixes == ('で/助動詞/で', 'す/語尾/す') or suffixes == ('られ/助動詞/られ', 'な/助動詞/な', 'い/語尾/い') or suffixes == ('か/語尾/か', 'れ/助動詞/れ', 'る/語尾/る') or suffixes == ('でき/動詞/でき', 'た/助動詞/た') or suffixes == ('ぐ/語尾/ぐ',) or suffixes == ('か/語尾/か', 'れ/助動詞/れ', 'る/語尾/る') or suffixes == ('で/助動詞/で', 'あ/動詞/あ', 'ろ/語尾/ろ', 'う/助動詞/う'):
      finishForm(head, suffixes)
   elif suffixes == ('い/語尾/い', 'だ/助動詞/だ') or suffixes == ('い/語尾/い', 'ま/助動詞/ま', 'し/語尾/し', 'た/助動詞/た') or suffixes == ('い/語尾/い', '始め/動詞/はじめ', 'る/語尾/る'): # this -i- belongs to the verb, e.g. 継いだ
      finishForm(head, suffixes)
   elif suffixes == ('な/助動詞/な', 'く/語尾/く', 'な/動詞/な', 'っ/語尾/っ', 'た/助動詞/た'): # 使えなくなった 'can no longer be used'. compare tsukaenai `cannot be used'
      finishForm(head, suffixes)
   elif suffixes == ('た/助動詞/た', 'し/動詞/し', 'た/助動詞/た'): # 来たした from 来す（きたす）-- the initial -ta- belongs to the verb
      finishForm(head, suffixes)
   elif suffixes == ('し/動詞/し', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('さ/動詞/さ', 'れ/助動詞/れ', 'な/助動詞/な', 'い/語尾/い') or suffixes == ('し/動詞/し', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('わ/語尾/わ', 'れ/助動詞/れ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('か/語尾/か', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('ら/語尾/ら', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('し/語尾/し', '始め/動詞/はじめ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('ら/語尾/ら', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('ま/助動詞/ま', 'せ/語尾/せ', 'ん/助動詞/ん'):
      finishForm(head, suffixes)
   elif suffixes == ('か/語尾/か', 'れ/助動詞/れ', 'な/助動詞/な', 'く/語尾/く', 'な/動詞/な', 'っ/語尾/っ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('だっ/助動詞/だっ', 'た/助動詞/た', 'らし/助動詞/らし', 'い/語尾/い'):
      finishForm(head, suffixes)
   elif suffixes == ('し/動詞/し', 'た/助動詞/た', 'い/語尾/い'):
      finishForm(head, suffixes)
   elif suffixes == ('ら/語尾/ら', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た'): # 終らなかった owo...   from owaru
      finishForm(head, suffixes)
   elif suffixes == ('り/語尾/り', '始め/動詞/はじめ', 'た/助動詞/た'): # 探/動詞/さぐ        ('り/語尾/り', '始め/動詞/はじめ', 'た/助動詞/た')
      finishForm(head, suffixes)
   elif suffixes == ('き/語尾/き', '始め/動詞/はじめ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif set(suffixes).issubset(affixMorphemes) and max([ord(x) for x in head[:head.index("/")]]) > 12400:
      finishForm(head, suffixes)
   elif (head == 'な/動詞/な' or head == 'な/形容詞/な') and set(suffixes).issubset(affixMorphemes):
      finishForm(head, suffixes)
   elif head == 'さ/動詞/さ' and set(suffixes).issubset(affixMorphemes):
      finishForm(head, suffixes)
   elif head == 'で/助動詞/で' and set(suffixes).issubset(affixMorphemes):
      finishForm(head, suffixes)
   elif head == 'い/動詞/い' and set(suffixes).issubset(affixMorphemes):
      finishForm(head, suffixes)
   elif head == 'し/動詞/し' and set(suffixes).issubset(affixMorphemes):
      finishForm(head, suffixes)
   elif head == 'あ/動詞/あ' and set(suffixes).issubset(affixMorphemes):
      finishForm(head, suffixes)
   elif 'て/助詞/て' in suffixes: # not included
#      print("WITH TE   ", head, "\t", suffixes)
      _ = 0
   else:
     print("....", head, "\t", suffixes)
   
counter = 0
with gzip.open("/u/scr/mhahn/FAIR18/WIKIPEDIA/japanese/japanese-train-tagged.txt.gz", mode="r") as inFile:
   for line in inFile:
      if formsCounter > MAX_FORMS:
          break
      line = line.decode().split(" ")
#      print(line)
      verb = []
      lastWord = None
      for word in line:
         try:
            form, pos, lemma = word.split("/")
         except ValueError:
            continue
         if pos == "動詞" or pos == "助動詞" or pos == "語尾" or word == "て/助詞/て":
             verb.append(word)
             if len(verb) == 1 and lastWord is not None:
               if lastWord[2] == "名詞" and word == "し/動詞/し":
                  verb.insert(0, lastWord[0])
               elif lastWord[0] == "を/助詞/を" or lastWord[0] == "が/助詞/が" or lastWord[0] == "に/助詞/に" or lastWord[0] == 'も/助詞/も' or lastWord[0] in ['から/助詞/から', 'の/助詞/の']:
                  _ = 0
               elif lastWord[2] == '助詞':
                 _ = 0
               elif lastWord[2] == '副詞':
                 _ = 0
               elif lastWord[2] == '接尾辞':
                 _ = 0
               elif lastWord[2] == '補助記号':
                 _ = 0
               elif word == 'で/助動詞/で' and lastWord[2] == '名詞':
                 _ = 0
               else:
                  verb.insert(0, lastWord[0])
         elif word == "。/補助記号/。" and len(verb) > 0:

             if len(verb) > 2 and verb[0] == "な/形容詞/な" and verb[1] == "く/語尾/く" and max([ord(x) for x in verb[2][:verb[2].index("/")]]) > 12500: # naku means `without'
                del verb[0]
                del verb[0]
             elif len(verb) > 2 and verb[0] == "多/形容詞/おお" and verb[1] == "く/語尾/く" and max([ord(x) for x in verb[2][:verb[2].index("/")]]) > 12500: # ooku means `a lot'
                del verb[0]
                del verb[0]
             elif (len(verb) > 1 and verb[1] in ['し/動詞/し', 'する/動詞/する']): # form with suru
                processVerb(verb)
             elif ("/動詞/" in verb[0]): # starts with verb
                processVerb(verb)
             elif ("/助動詞/" in verb[0]): # or with 'auxiliary'
                processVerb(verb)
             elif ("/形容詞/" in verb[0] and verb[1] == 'い/語尾/い' and len(verb) == 2): # adjective forms with -i
                processVerb(verb)
             elif "/名詞/" in verb[0] and verb[1] in ['さ/動詞/さ', 'でき/動詞/でき', 'だっ/助動詞/だっ', 'で/助動詞/で', 'し/語尾/し', 'だ/助動詞/だ']: # noun + sa...
                processVerb(verb)
             elif "/形状詞/" in verb[0] and verb[1] in ['で/助動詞/で', 'かっ/語尾/かっ',]: # noun + sa...
                processVerb(verb)
             elif "/形状詞/" in verb[0] and verb[1] in ['かっ/語尾/かっ',]: # noun + sa...
                processVerb(verb)
             elif "/形状詞/" in verb[0]:
                processVerb(verb)
             elif ("/形容詞/" in verb[0]): 
                processVerb(verb)
             elif len(verb) > 1 and max([ord(x) for x in verb[1][:verb[1].index("/")]]) > 12500: # these should not be counted: the second element contains Kanji, so is likely a content morpheme
                if verb[1] == "出来/動詞/でき" and set(verb[2:]).issubset(affixMorphemes):
                   processVerb(verb)
                elif "/名詞/" in verb[0] and "/動詞/" in verb[1]: # noun + verb, just remove the noun 
                  del verb[0]
                  processVerb(verb)
                else:
                   print("===========KANJI=========", "\t".join(verb))
                _ = 0
             elif 'て/助詞/て' in verb: # not included
#               print("WITH TE TODO ", verb)
               _ = 0
             else:
               print("TODO", verb)
               _ = 0
             verb = []
         elif len(verb) > 0:
             verb = []
         lastWord = (word, form, pos, lemma)


from collections import defaultdict

pairs = defaultdict(int)

#print(forms_data)

chains = defaultdict(int)
morphemes = defaultdict(int)

for verb, suffixes in forms_data:
   chains[tuple(suffixes)] += 1
   for i in range(len(suffixes)):
      atI = suffixes[i]
      morphemes[atI] += 1
      for j in range(i):
         atJ = suffixes[j]
         pairs[(atI, atJ)] += 1
         if (atJ, atI) in pairs and pairs[(atJ, atI)] >= pairs[(atI, atJ)]:
            print("CONFLICT", atI, atJ, verb, suffixes)

chains = sorted([(x, y) for x, y in chains.items()], key=lambda x:x[1], reverse=True)

print(chains)

for i in range(min(len(chains), 100)):
  print(chains[i])

print(morphemes)
#print(pairs)



words = []

data = []

for verb, suffixes in forms_data:
  data.append([verb] + list(suffixes))

words = []

affixFrequency = {}
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixLemma = affix
    affixFrequency[affixLemma] = affixFrequency.get(affixLemma, 0)+1


itos = set()
for verbWithAff in data:
  for affix in verbWithAff[1:]:
    affixLemma = affix
    itos.add(affixLemma)
itos = sorted(list(itos))
stoi = dict(list(zip(itos, range(len(itos)))))


print(itos)
print(stoi)

itos_ = itos[::]
shuffle(itos_)
weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))

fullToWordD = {}
def fullToWord(x):
   if x not in fullToWordD:
      fullToWordD[x] = x[:x.index("/")]
   return fullToWordD[x]

def calculateTradeoffForWeights(weights):
    dev = []
    for verb in data:
       affixes = verb[1:]
       affixes = sorted(affixes, key=lambda x:weights[x])
       for ch in [verb[0]] + affixes:
         for char in fullToWord(ch):
           dev.append(char)
       #    print(char)
       dev.append("EOS")
       for _ in range(args.cutoff+2):
         dev.append("PAD")
       dev.append("SOS")
    
    itos = list(set(dev))
    
    
    dev = dev[::-1]
    #dev = list(createStreamContinuous(corpusDev))[::-1]
    
    
    #corpusTrain = CorpusIterator(args.language,"dev", storeMorph=True).iterator(rejectShortSentences = False)
    #train = list(createStreamContinuous(corpusTrain))[::-1]
    train = dev
    
    idev = range(len(dev))
    itrain = range(len(train))
    
    idev = sorted(idev, key=lambda i:dev[i:i+20])
    itrain = sorted(itrain, key=lambda i:train[i:i+20])
    
#    print(idev)
    
    idevInv = [x[1] for x in sorted(zip(idev, range(len(idev))), key=lambda x:x[0])]
    itrainInv = [x[1] for x in sorted(zip(itrain, range(len(itrain))), key=lambda x:x[0])]
    
    assert idev[idevInv[5]] == 5
    assert itrain[itrainInv[5]] == 5
    
    
    
    def getStartEnd(k):
       start = [0 for _ in dev]
       end = [len(train)-1 for _ in dev]
       if k == 0:
          return start, end
       # Start is the FIRST train place that is >=
       # End is the FIRST train place that is >
       l = 0
       l2 = 0
       for j in range(len(dev)):
         prefix = tuple(dev[idev[j]:idev[j]+k])
         while l2 < len(train):
            prefix2 = tuple(train[itrain[l2]:itrain[l2]+k])
            if prefix <= prefix2:
                 start[j] = l2
                 break
            l2 += 1
         if l2 == len(train):
            start[j] = l2
         while l < len(train):
            prefix2 = tuple(train[itrain[l]:itrain[l]+k])
            if prefix < prefix2:
                 end[j] = l
                 break
            l += 1
         if l == len(train):
            end[j] = l
         start2, end2 = start[j], end[j]
         assert start2 <= end2
         if start2 > 0 and end2 < len(train):
           assert prefix > tuple(train[itrain[start2-1]:itrain[start2-1]+k])
           assert prefix <= tuple(train[itrain[start2]:itrain[start2]+k])
           assert prefix >= tuple(train[itrain[end2-1]:itrain[end2-1]+k])
           assert prefix < tuple(train[itrain[end2]:itrain[end2]+k])
       return start, end
    
    
    lastProbability = [None for _ in idev]
    newProbability = [None for _ in idev]
    
    devSurprisalTable = []
    for k in range(0,args.cutoff):
#       print(k)
       startK, endK = getStartEnd(k) # Possible speed optimization: There is some redundant computation here, could be reused from the previous iteration. But the algorithm is very fast already.
       startK2, endK2 = getStartEnd(k+1)
       cachedFollowingCounts = {}
       for j in range(len(idev)):
    #      print(dev[idev[j]])
          if dev[idev[j]] in ["PAD", "SOS"]:
             continue
          start2, end2 = startK2[j], endK2[j]
          devPref = tuple(dev[idev[j]:idev[j]+k+1])
          if start2 > 0 and end2 < len(train):
            assert devPref > tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]), (devPref, tuple(train[itrain[start2-1]:itrain[start2-1]+k+1]))
            assert devPref <= tuple(train[itrain[start2]:itrain[start2]+k+1]), (devPref, tuple(train[itrain[start2]:itrain[start2]+k+1]))
            assert devPref >= tuple(train[itrain[end2-1]:itrain[end2-1]+k+1])
            assert devPref < tuple(train[itrain[end2]:itrain[end2]+k+1])
    
          assert start2 <= end2
    
          countNgram = end2-start2
          if k >= 1:
             if idev[j]+1 < len(idevInv):
               prefixIndex = idevInv[idev[j]+1]
               assert dev[idev[prefixIndex]] == dev[idev[j]+1]
       
               prefixStart, prefixEnd = startK[prefixIndex], endK[prefixIndex]
               countPrefix = prefixEnd-prefixStart
               if countPrefix < args.gamma: # there is nothing to interpolate with, just back off
                  assert k > 0
                  newProbability[j] = lastProbability[j]
               else:
                  assert countPrefix >= countNgram, (countPrefix, countNgram)
       
                  following = set()
                  if (prefixStart, prefixEnd) in cachedFollowingCounts:
                      followingCount = cachedFollowingCounts[(prefixStart, prefixEnd)]
                  else:
                    for l in range(prefixStart, prefixEnd):
                      if k < itrain[l]+1:
                         following.add(train[itrain[l]-1])
                         assert devPref[1:] == tuple(train[itrain[l]-1:itrain[l]+k])[1:], (k, itrain[l], l, devPref , tuple(train[itrain[l]-1:itrain[l]+k]))
                    followingCount = len(following)
                    cachedFollowingCounts[(prefixStart, prefixEnd)] = followingCount
                  if followingCount == 0:
                      newProbability[j] = lastProbability[j]
                  else:
                      assert countNgram > 0
                      probability = log(max(countNgram - args.alpha, 0.0) + args.alpha * followingCount * exp(lastProbability[j])) -  log(countPrefix)
                      newProbability[j] = probability
             else:
                newProbability[j] = lastProbability[j]
          elif k == 0:
                  probability = log(countNgram + args.delta) - log(len(train) + args.delta * len(itos))
                  newProbability[j] = probability
       lastProbability = newProbability 
       newProbability = [None for _ in idev]
       assert all([x is None or x <=0 for x in lastProbability])
       try:
           lastProbabilityFiltered = [x for x in lastProbability if x is not None]
           surprisal = - sum([x for x in lastProbabilityFiltered])/len(lastProbabilityFiltered)
       except ValueError:
    #       print >> sys.stderr, "PROBLEM"
     #      print >> sys.stderr, lastProbability
           surprisal = 1000
       devSurprisalTable.append(surprisal)
     #  print("Surprisal", surprisal, len(itos))
    #print(devSurprisalTable)
    mis = [devSurprisalTable[i] - devSurprisalTable[i+1] for i in range(len(devSurprisalTable)-1)]
    tmis = [mis[x]*(x+1) for x in range(len(mis))]
    #print(mis)
    #print(tmis)
    auc = 0
    memory = 0
    mi = 0
    for i in range(len(mis)):
       mi += mis[i]
       memory += tmis[i]
       auc += mi * tmis[i]
    #print("MaxMemory", memory)
    assert 7>memory
    auc += mi * (7-memory)
    #print("AUC", auc)
    return auc
    #assert False
    
    #outpath = TARGET_DIR+"/estimates-"+args.language+"_"+__file__+"_model_"+str(myID)+"_"+args.model+".txt"
    #print(outpath)
    #with open(outpath, "w") as outFile:
    #         print >> outFile, str(args)
    #         print >> outFile, devSurprisalTable[-1]
    #         print >> outFile, " ".join(map(str,devSurprisalTable))
    #
    #
   
from random import random

for iteration in range(1000):
  coordinate=choice(itos)
  while affixFrequency.get(coordinate, 0) < 10 and random() < 0.95:
     coordinate = choice(itos)
  mostCorrect, mostCorrectValue = 0, None
  for newValue in [-1] + [2*x+1 for x in range(len(itos))] + [weights[coordinate]]:
     if random() < 0.8 and newValue != weights[coordinate]:
        continue
     print(newValue, mostCorrect, coordinate, affixFrequency[coordinate])
     weights_ = {x : y if x != coordinate else newValue for x, y in weights.items()}
     correctCount = calculateTradeoffForWeights(weights_)
#     print(weights_)
#     print(coordinate, newValue, iteration, correctCount)
     if correctCount > mostCorrect:
        mostCorrectValue = newValue
        mostCorrect = correctCount
  print(iteration, mostCorrect)
  weights[coordinate] = mostCorrectValue
  itos_ = sorted(itos, key=lambda x:weights[x])
  weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))
  print(weights)
  for x in itos_:
     if affixFrequency[x] < 10:
       continue
     print("\t".join([str(y) for y in [x, weights[x], affixFrequency[x]]]))
  if (iteration + 1) % 50 == 0:
     with open(TARGET_DIR+"/optimized_"+__file__+"_"+str(myID)+".tsv", "w") as outFile:
        print(iteration, mostCorrect, str(args), file=outFile)
        for key in itos_:
           print(key, weights[key], file=outFile)



