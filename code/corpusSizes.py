# ./python27 corpusSizes.py > corpusSizes.tsv


from ud_languages import languages

from corpusIterator import CorpusIterator

with open("corpusSizes.tsv", "w") as outFile:
  print >> outFile, ("Language\tTrainingSents")
  for language in languages:
    train = [x for x in CorpusIterator(language,"train", storeMorph=False).iterator()]
    print >> outFile, (language+"\t"+str( len(train)))

