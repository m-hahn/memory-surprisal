library(tidyr)
library(dplyr)
library(ggplot2)


fullData = read.csv("../../../results/tradeoff/listener-curve-histogram_byMem.tsv", sep="\t")

memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    dataL = read.csv(paste("../../../results/raw/word-level/",language,"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", sep=""), sep="\t")
    UnigramCE = mean(dataL$UnigramCE)
    data = fullData %>% filter(Language == language)
    barWidth = (max(data$MI) - min(data$MI))/30
    plot = ggplot(data, aes(x=UnigramCE-MI, fill=Type, color=Type)) + theme_classic() + theme(legend.position="none")   + geom_density(data= data%>%filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled..))      + geom_bar(data = data %>% filter(Type %in% c("REAL_REAL", "GROUND")) %>% group_by(Type) %>% summarise(MI=mean(MI)) %>% mutate(y=1),  aes(y=y, group=Type), width=barWidth, stat="identity", position = position_dodge()) 
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_byMem_onlyWordForms_boundedVocab.pdf", sep=""))
    return(plot)
}

languages = read.csv("languages.tsv", sep="\t")

for(language in languages$Language) {
   memListenerSurpPlot_onlyWordForms_boundedVocab(language)
}

