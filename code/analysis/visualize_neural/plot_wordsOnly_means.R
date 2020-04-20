library(tidyr)
library(dplyr)
library(ggplot2)


fullData = read.csv("../../../results/tradeoff/listener-curve-means.tsv", sep="\t")

memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    dataL = read.csv(paste("../../../results/raw/word-level/",language,"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", sep=""), sep="\t")
    UnigramCE = mean(dataL$UnigramCE)
    data = fullData %>% filter(Language == language)
    plot = ggplot(data, aes(x=Memory, y=UnigramCE-MI_Mean, fill=Type, color=Type)) + geom_line(size=2)+ theme_classic() + theme(legend.position="none") + geom_line(aes(x=Memory, y=UnigramCE-MI_Mean-MI_SD), linetype="dashed") + geom_line(aes(x=Memory, y=UnigramCE-MI_Mean+MI_SD), linetype="dashed")
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEANS_onlyWordForms_boundedVocab.pdf", sep=""))
    return(plot)
}

languages = read.csv("languages.tsv", sep="\t")

for(language in languages) {
   memListenerSurpPlot_onlyWordForms_boundedVocab(language)
}

