import gzip

formsCounter = 0

def finishForm(head, suffixes):
  global formsCounter
  formsCounter += 1
  print(formsCounter)



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
   elif suffixes == ('い/語尾/い', 'だ/助動詞/だ'): # this -i- belongs to the verb, e.g. 継いだ
      finishForm(head, suffixes)
   elif suffixes == ('た/助動詞/た', 'し/動詞/し', 'た/助動詞/た'): # 来たした from 来す（きたす）-- the initial -ta- belongs to the verb
      finishForm(head, suffixes)
   elif suffixes == ('し/動詞/し', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif suffixes == ('さ/動詞/さ', 'れ/助動詞/れ', 'な/助動詞/な', 'い/語尾/い') or suffixes == ('し/動詞/し', 'な/助動詞/な', 'かっ/語尾/かっ', 'た/助動詞/た'):
      finishForm(head, suffixes)
   elif 'て/助詞/て' in suffixes: # not included
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
               print("TODO", verb)
               _ = 0
             verb = []
         elif len(verb) > 0:
             verb = []
         lastWord = (word, form, pos, lemma)
