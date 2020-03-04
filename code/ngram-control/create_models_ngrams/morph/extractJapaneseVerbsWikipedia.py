import gzip

def processVerb(verb):
   head = verb[0]
   suffixes = tuple(verb[1:])
   if suffixes == ('る/語尾/る',):
      _ = 0
   elif suffixes == ('た/助動詞/た',):
      _ = 0
   elif suffixes == ('う/語尾/う',):
      _ = 0
   elif suffixes == ('い/語尾/い',):
      _ = 0
   elif suffixes == ('し/動詞/し', 'た/助動詞/た'):
      _ = 0
   else:
     print("....", head, "\t", suffixes)
   
counter = 0
with gzip.open("/u/scr/mhahn/FAIR18/WIKIPEDIA/japanese/japanese-train-tagged.txt.gz", mode="r") as inFile:
   for line in inFile:
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
             elif len(verb) > 1 and max([ord(x) for x in verb[1][:verb[1].index("/")]]) > 12400: # these should not be counted: the second element contains Kanji, so is likely a content morpheme
#                print("===========", verb)
                _ = 0
             else:
               print(verb)
             verb = []
         elif len(verb) > 0:
             verb = []
         lastWord = (word, form, pos, lemma)
