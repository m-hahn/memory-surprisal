
# Plots medians with confidence intervals

fullData = read.csv("../../../../results/tradeoff/ngrams/listener-curve-ci-median.tsv", sep="\t")

memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    dataL = read.csv(paste("../../../../results/raw/ngrams/",language,"_ngrams_decay_after_tuning.tsv", sep=""), sep="\t")
    UnigramCE = mean(dataL$UnigramCE)
    data = fullData %>% filter(Language == language)
    plot = ggplot(data, aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianEmpirical), fill=Type, color=Type)) + geom_line(size=2)+ theme_classic() + theme(legend.position="none") + geom_line(aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianLower)), linetype="dashed") + geom_line(aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianUpper)), linetype="dashed")
    plot = plot + xlab("Memory") + ylab("Median Surprisal")
    plot = plot + theme(text = element_text(size=30))
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIANS_onlyWordForms_boundedVocab.pdf", sep=""))
    return(plot)
}


languages = read.csv("../../../corpus_size/corpusSizes.tsv", sep="\t")
languages = languages$Language

for(language in languages) {
   memListenerSurpPlot_onlyWordForms_boundedVocab(language)
}


