library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

fullData = read.csv("../../../results/tradeoff/listener-curve-histogram_byMem.tsv", sep="\t") %>% filter(Type != "GROUND")

memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    dataL = read.csv(paste("../../../results/raw/word-level/",language,"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", sep=""), sep="\t")
    UnigramCE = mean(dataL$UnigramCE)
    data = fullData %>% filter(Language == language)
    barWidth = (max(data$MI) - min(data$MI))/30
    plot = ggplot(data, aes(x=UnigramCE-MI, fill=Type, color=Type)) + theme_classic() + theme(legend.position="none")   + geom_density(data= data%>%filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled..))      + geom_bar(data = data %>% filter(Type %in% c("REAL_REAL", "GROUND")) %>% group_by(Type) %>% summarise(MI=mean(MI)) %>% mutate(y=1),  aes(y=y, group=Type), width=barWidth, stat="identity", position = position_dodge()) 


    plot = plot + ylab("Density")
    plot = plot + xlab("Surprisal")

    plot = plot + theme(axis.text.x = element_text(size=20))  
    plot = plot + theme(axis.text.y = element_text(size=20))  
    plot = plot + theme(axis.title.x = element_text(size=20))  
    plot = plot + theme(axis.title.y = element_text(size=20))  


    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_byMem_onlyWordForms_boundedVocab_REAL.pdf", sep=""), height=3, width=6)

    return(plot)
}

languages = read.csv("languages.tsv", sep="\t")

for(language in languages$Language) {
   memListenerSurpPlot_onlyWordForms_boundedVocab(language)
}

