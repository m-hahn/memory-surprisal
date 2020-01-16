
#    tamw contains documents annotated on all three annotation layers (morphological, analytical, tectogrammatical, together with all additional annotation and all corrections done after PDT 2.0 has been released),

import gzip

# Of interest: tfa
import os
import sys


import xml.etree.ElementTree as ET

def recursivelyBuildTree(lm, SENT_ID):
   result = {"id" : lm.attrib["id"], "children" : [], "tfa" : "NULL", "t_lemma" : "NULL"}
   for child in lm:
      if child.tag.endswith("children"):
          for child2 in child:
              result["children"].append(recursivelyBuildTree(child2, SENT_ID))
      elif child.tag.endswith("tfa"):
        result["tfa"] = child.text
      elif child.tag.endswith("t_lemma"):
        result["t_lemma"] = child.text
   wordsToData[SENT_ID][result["id"][2:]] = result
   return result

wordsToData = {}


for directory in os.listdir("/u/scr/mhahn/CORPORA/czech_pdt/PDT3.5/data/tamw/"):
  BASE_DIR = "/u/scr/mhahn/CORPORA/czech_pdt/PDT3.5/data/tamw/"+directory
  files = sorted(list(set([x[:-5] for x in os.listdir(BASE_DIR)])))
  print(files)
  for fileName in files:
    print(fileName) 
    tree_m = ET.fromstring(gzip.open(BASE_DIR+"/"+fileName+".m.gz", "r").read())
    tree_t = ET.fromstring(gzip.open(BASE_DIR+"/"+fileName+".t.gz", "r").read())
    
    for child in tree_t:
       if child.tag.endswith("trees"):
         for sentence in child:
            SENT_ID = sentence.attrib["id"][2:]
            wordsToData[SENT_ID] = {}
            for annotation in sentence:
                if annotation.tag.endswith("children"):
                    for lm in annotation:
                      fullTree = recursivelyBuildTree(lm, SENT_ID) #print("LM", lm.attrib)
    
    
    # c contrastive & bound
    # f non-bound
    # t non-contrastive bound
    
    for child in tree_m: 
       if child.tag.endswith("s"):
           SENT_ID = child.attrib["id"][2:]
           sentence = []
           wordsToData[SENT_ID]["linearized"] = sentence
           for child2 in child:
               dataHere = wordsToData[SENT_ID].get(child2.attrib["id"][2:], {})
               wordsToData[SENT_ID][child2.attrib["id"][2:]] = dataHere
               wordForm = None
               for anno in child2:
                   if anno.tag.endswith("form"):
                      dataHere["wordForm"] = anno.text
                      wordForm = anno.text
                      break
               sentence.append((wordForm, dataHere))
           surfaceString = " ".join([x[0] for x in sentence])

counter = 0
for partition in ["test", "dev", "train"]:
  with open("/u/scr/mhahn/CORPORA/czech_pdt_infostruc/"+partition+".conllu", "w") as outFile:
   with open("/u/scr/corpora/Universal_Dependencies/Universal_Dependencies_2.4/ud-treebanks-v2.4/UD_Czech-PDT/cs_pdt-ud-"+partition+".conllu", "r") as inFile:
      sentence = []
      attribs = {}
      for line in inFile:
        line = line.strip()
        if line.startswith("# "):
           print(line, file=outFile)
           try:
              index = line.index("=")
              attribs[line[:index-1]] = line[index+2:]
           except ValueError:
            _ = 0
        elif len(line) <= 1:
          if len(attribs) > 0:
             counter += 1
             sentenceID = attribs['# sent_id']
             if sentenceID in wordsToData:
                annotated = wordsToData[sentenceID]["linearized"]
                fromAnnotation = [x[0] for x in annotated]
                fromUD = attribs["# text"]
                if "".join(fromAnnotation) != fromUD.replace(" ", ""):
                  print((fromAnnotation, fromUD), file=sys.stderr)
                print(fromAnnotation)
                lastWord = sentence[-1]
                counterInAnnotation = 0
                counterInUD = 0
                while counterInUD < len(sentence):
                   line = sentence[counterInUD]
                   line = line.split("\t")
                   print(line)
                   print(counterInUD, counterInAnnotation)
                   form = line[1]
                   if form != fromAnnotation[counterInAnnotation]:
                     print((form, fromAnnotation[counterInAnnotation]), file=sys.stderr)
                   sentence[counterInUD]+="\t"+annotated[counterInAnnotation][1].get("tfa", "N")
                   print(line[0], "-" in line[0])

                   # One token in PDT corresponding to a sequence in the UD version
                   if "-" in line[0]:
                      start, end = line[0].split("-")
                      for i in range(counterInUD+1, counterInUD + 2 + int(end) - int(start)):
                         sentence[i]+="\t"+annotated[counterInAnnotation][1].get("tfa", "N")
                      counterInUD += 2 + int(end) - int(start)
                   else:
                      counterInUD += 1
                   counterInAnnotation += 1
             else:
                print(("MISSING SENTENCE", sentenceID), file=sys.stderr)
                for i in range(len(sentence)):
                    sentence[i] += "\tUNK"

             for line in sentence:
                assert line[-1] in ["N", "t", "f", "c"] or line.endswith("UNK") or line.endswith("NULL"), line
                assert "\n" not in line
                print(line, file=outFile)

          print("\n\n", file=outFile)

          sentence = []
          attribs = {}
        else:
          sentence.append(line)
          
