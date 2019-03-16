# ./python27 corpusSizes.py > corpusSizes.tsv


from corpusIterator import CorpusIterator

from ud_languages import languages


with open("../corpusSizes.tsv", "w") as outFile:
  print >> outFile, ("Language\tTrainingSents\tHeldoutSents")
  for language in languages:
    train = [x for x in CorpusIterator(language,"train", storeMorph=False).iterator()]
    heldout = [x for x in CorpusIterator(language,"dev", storeMorph=False).iterator()]

    print >> outFile, (language+"\t"+str( len(train))+"\t"+str(len(heldout)))

