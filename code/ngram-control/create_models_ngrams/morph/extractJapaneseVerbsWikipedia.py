import gzip
from random import shuffle, choice

formsCounter = 0


forms_data = []

def finishForm(head, suffixes):
  global formsCounter
  formsCounter += 1
  if formsCounter % 100 == 0:
     print(formsCounter)
  forms_data.append((head, suffixes))
    
affixMorphemes = set(['き/語尾/き', '始め/動詞/はじめ', 'た/助動詞/た', 'る/語尾/る', 'な/動詞/な', 'っ/語尾/っ', 'でき/動詞/でき', 'な/助動詞/な', 'く/語尾/く', 'な/動詞/な', 'る/語尾/る', 'さ/動詞/さ', 'れ/助動詞/れ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'み/語尾/み', '始め/動詞/はじめ', 'た/助動詞/た', 'が/語尾/が', 'れ/助動詞/れ', 'た/助動詞/た', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'だ/助動詞/だ', 'ろ/語尾/ろ', 'う/助動詞/う', 'わ/語尾/わ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'ば/語尾/ば', 'な/助動詞/な', 'い/語尾/い', 'ら/語尾/ら', 'せ/助動詞/せ', 'た/助動詞/た', 'られ/助動詞/られ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'られ/助動詞/られ', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た', 'く/語尾/く', 'だ/助動詞/だ', 'ろ/語尾/ろ', 'う/助動詞/う'])

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
   elif 'て/助詞/て' in suffixes: # not included
      _ = 0
   else:
     print("....", head, "\t", suffixes)
   
counter = 0
with gzip.open("/u/scr/mhahn/FAIR18/WIKIPEDIA/japanese/japanese-train-tagged.txt.gz", mode="r") as inFile:
   for line in inFile:
      if formsCounter > 100000:
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
             if (len(verb) > 1 and verb[1] in ['し/動詞/し', 'する/動詞/する']): # form with suru
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
             elif len(verb) > 1 and max([ord(x) for x in verb[1][:verb[1].index("/")]]) > 12500: # these should not be counted: the second element contains Kanji, so is likely a content morpheme
                if verb[1] == "出来/動詞/でき" and set(verb[2:]).issubset(affixMorphemes):
                   processVerb(verb)
                elif "/名詞/" in verb[0] and "/動詞/" in verb[1]: # noun + verb, just remove the noun 
                  del verb[0]
                  processVerb(verb)
                else:
                   print("===========KANJI=========", "\t".join(verb))
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
            print(atI, atJ, verb, suffixes)

chains = sorted([(x, y) for x, y in chains.items()], key=lambda x:x[1], reverse=True)

print(chains)

for i in range(min(len(chains), 100)):
  print(chains[i])

print(morphemes)
#print(pairs)



words = []

data = forms_data

affixFrequencies = {}
for verb, suffixes in data:
  for affix in suffixes:
    affixFrequencies[affix] = affixFrequencies.get(affix, 0)+1

dataPerAffixes = defaultdict(list)

itos = set()
for verb, suffixes in data:
  for affix in suffixes:
    itos.add(affix)
    dataPerAffixes[affix].append((verb, suffixes))
    dataPerAffixes[None].append((verb, suffixes))
itos = sorted(list(itos))
stoi = dict(list(zip(itos, range(len(itos)))))


print(itos)
print(stoi)

itos_ = itos[::]
shuffle(itos_)
weights = dict(list(zip(itos_, [2*x for x in range(len(itos_))])))

def getCorrectOrderCount(weights, coordinate, newValue):
   correct = 0
   incorrect = 0
   for verb, suffixes in dataPerAffixes[coordinate]:
      for i in range(0, len(suffixes)):
         for j in range(0, i):
             if suffixes[i] == coordinate:
                 weightI = newValue
             else:
                weightI = weights[suffixes[i]]

             if suffixes[j] == coordinate:
                 weightJ = newValue
             else:
                weightJ = weights[suffixes[j]]
             if weightI > weightJ:
               correct+=1
             else:
               incorrect+=1
   if correct+incorrect == 0:
      return 1.0
   return correct/(correct+incorrect)

for iteration in range(200):
  coordinate = choice(itos)
  mostCorrect, mostCorrectValue = 0, None
  for newValue in [-1] + [2*x+1 for x in range(len(itos))]:
     correctCount = getCorrectOrderCount(weights, coordinate, newValue)
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
     print("\t".join([str(y) for y in [x, weights[x], affixFrequencies[x]]]))
  print(getCorrectOrderCount(weights, None, 0))

with open("output/extracted_"+__file__+"_"+str(myID)+".tsv", "w") as outFile:
  for x in itos_:
  #   if affixFrequencies[x] < 10:
   #    continue
     print("\t".join([str(y) for y in [x, weights[x], affixFrequencies[x]]]), file=outFile)






#し/動詞/し      0       14779   # suru derives verbs from (only Sino-Japanese?) nouns
#さ/語尾/さ      2       884	#
#する/語尾/する  4       322
#わ/語尾/わ      6       1482
#ぶ/語尾/ぶ      8       401
#か/語尾/か      10      311
#で/助動詞/で    12      825
#あ/動詞/あ      14      14788
#ん/語尾/ん      16      854
#だっ/助動詞/だっ        18      1154
#ぜ/語尾/ぜ      20      23
#ば/語尾/ば      22      665
#ま/助動詞/ま    24      25
#む/語尾/む      26      358
#ふ/語尾/ふ      28      2
#ま/語尾/ま      30      267
#する/動詞/する  32      5668
#み/語尾/み      34      16
#さ/動詞/さ      36      9876
#ら/語尾/ら      38      1138
#たが/助動詞/たが        40      1
#続け/動詞/つづけ        42      36
#しめ/助動詞/しめ        44      4
#ぐ/語尾/ぐ      46      54
#す/語尾/す      48      1275
#つ/語尾/つ      50      480
#せ/助動詞/せ    52      717
#し/語尾/し      54      2064
#せ/語尾/せ      56      19
#ん/助動詞/ん    58      19
#き/語尾/き      60      37
#じ/語尾/じ      62      524
#られ/助動詞/られ        64      2687
#り/語尾/り      66      18
#い/動詞/い      68      1
#始め/動詞/はじめ        70      195
#れ/助動詞/れ    72      13813
#な/助動詞/な    74      2496
#かっ/語尾/かっ  76      1177
#く/語尾/く      78      596
#な/動詞/な      80      117
#っ/語尾/っ      82      12082
#う/語尾/う      84      2371
#が/語尾/が      86      24
#らし/助動詞/らし        88      3
#い/語尾/い      90      5012
#た/助動詞/た    92      50444
#だ/助動詞/だ    94      1101
#ろ/語尾/ろ      96      27
#う/助動詞/う    98      28
#でき/動詞/でき  100     586
#る/語尾/る      102     32119
#0.9918453649784598


