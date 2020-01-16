import os
import random
import sys

header = ["index", "word", "lemma", "posUni", "posFine", "morph", "head", "dep", "_", "_", "infostruc"]

SMALL_TREEBANKS = ["UD_Kazakh-KTB", "UD_Cantonese-HK", "UD_Naija-NSC", "UD_Buryat-BDT", "UD_Thai-PUD", "UD_Breton-KEB", "UD_Faroese-OFT", "UD_Amharic-ATT", "UD_Kurmanji-MG", "UD_Upper_Sorbian-UFAL", "UD_Bambara-CRB", "UD_Erzya-JR"]
SUBSTITUTE_TEST_FOR_DEV = ["UD_North_Sami", "UD_Irish", "UD_Buryat-BDT", "UD_Armenian-ArmTDP"]
SUBSTITUTE_DEV_FOR_TRAIN = ["UD_Armenian-ArmTDP"]

def readUDCorpus(language, partition):
      assert language == "Czech-PDT"


      partitionHere = partition

         
      # Collect corpus files
      try:
         dataPath = "/u/scr/mhahn/CORPORA/czech_pdt_infostruc/"+partition+".conllu"
         with open(dataPath, "r") as inFile:
            data = inFile.read().strip().split("\n\n\n\n")
      except IOError:
         print >> sys.stderr, "Did not find "+dataPath
      assert len(data) > 0, (language, partition)

      data = [x for x in data if not x.endswith("UNK")]
      print >> sys.stderr, "Read "+str(len(data))+ " sentences "+partition+" datasets. "
      return data

class CorpusIterator():
   def __init__(self, language, partition="train", storeMorph=False, splitLemmas=False, shuffleData=True, shuffleDataSeed=None, splitWords=False):
      if splitLemmas:
           assert language == "Korean"
      self.splitLemmas = splitLemmas
      self.splitWords = splitWords
      assert self.splitWords == (language == "BKTreebank_Vietnamese")

      self.storeMorph = storeMorph
      if language == "BKTreebank_Vietnamese":
          import accessBKTreebank
          data = accessBKTreebank.readBKTreebank(partition)
      elif language == "TuebaJS":
         import accessTuebaJS
         data = accessTuebaJS.readTuebaJSTreebank(partition)
         assert len(data) > 0, (language, partition)
      elif language == "LDC2012T05":
         import accessChineseDependencyTreebank
         data = accessChineseDependencyTreebank.readChineseDependencyTreebank(partition)
         assert len(data) > 0, (language, partition)
        
      else:
          data = readUDCorpus(language, partition)
      if shuffleData:
       if shuffleDataSeed is None:
         random.shuffle(data)
       else:
         random.Random(shuffleDataSeed).shuffle(data)

      self.data = data
      self.partition = partition
      self.language = language
      assert len(data) > 0, (language, partition)
   def permute(self):
      random.shuffle(self.data)
   def length(self):
      return len(self.data)
   def processSentence(self, sentence):
        sentence = map(lambda x:x.split("\t"), sentence.split("\n"))
        result = []
        for i in range(len(sentence)):
#           print sentence[i]
           if sentence[i][0].startswith("#"):
              continue
           if "-" in sentence[i][0]: # if it is NUM-NUM
              continue
           if "." in sentence[i][0]:
              continue
 #          print(header)
#           print(sentence[i])
           if len(sentence[i]) < len(header):
              print(self.partition, "ERROR", sentence[i])
           while len(sentence[i]) < len(header):
              sentence[i].append("_")
           sentence[i] = dict([(y, sentence[i][x]) for x, y in enumerate(header)])
           sentence[i]["head"] = int(sentence[i]["head"])
           sentence[i]["index"] = int(sentence[i]["index"])
           sentence[i]["word"] = sentence[i]["word"].lower()
           if sentence[i]["infostruc"] == "NULL":
                sentence[i]["infostruc"] = "N"
           if self.language == "Thai-Adap":
              assert sentence[i]["lemma"] == "_"
              sentence[i]["lemma"] = sentence[i]["word"]
           if self.splitLemmas:
              sentence[i]["lemmas"] = sentence[i]["lemma"].split("+")

           if self.storeMorph:
              sentence[i]["morph"] = sentence[i]["morph"].split("|")

           if self.splitWords:
              sentence[i]["words"] = sentence[i]["word"].split("_")


           sentence[i]["dep"] = sentence[i]["dep"].lower()
           if self.language == "LDC2012T05" and sentence[i]["dep"] == "hed":
              sentence[i]["dep"] = "root"
           if self.language == "LDC2012T05" and sentence[i]["dep"] == "wp":
              sentence[i]["dep"] = "punct"

           result.append(sentence[i])
 #          print sentence[i]
        return result
   def getSentence(self, index):
      result = self.processSentence(self.data[index])
      return result
   def iterator(self, rejectShortSentences = False):
     for sentence in self.data:
        if len(sentence) < 3 and rejectShortSentences:
           continue
        yield self.processSentence(sentence)

if __name__ == "__main__":
   data = CorpusIterator("Czech-PDT", "dev")
   data = CorpusIterator("Czech-PDT", "train")
   data = CorpusIterator("Czech-PDT", "test")
