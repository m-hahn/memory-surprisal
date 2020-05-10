# ./python27 corpusSizes.py > corpusSizes.tsv

# Result extracted from tex file:
# c(1315,974,21864,514,926,5396,788,8907,808,550,13123,3997,7689,102993,4383,18310,17062,1450,6959,1108,27198,32347,13814,1662,5241,13304,910,4477,17427,7164,947,27410,634,4124,1123,848,2257,29870,4798,6100,17995,8664,52664,2935,8483,7532,28492,7041,900,3685,4506,4043,1656,1400)



from corpusIterator import CorpusIterator

from ud_languages import languages


with open("../corpusSizes.tsv", "w") as outFile:
  print >> outFile, ("Language\tTrainingSents\tHeldoutSents")
  for language in languages:
    train = [x for x in CorpusIterator(language,"train", storeMorph=False).iterator()]
    heldout = [x for x in CorpusIterator(language,"dev", storeMorph=False).iterator()]

    print >> outFile, (language+"\t"+str( len(train))+"\t"+str(len(heldout)))

