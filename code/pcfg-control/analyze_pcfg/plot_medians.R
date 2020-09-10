library(tidyr)
library(dplyr)
library(ggplot2)

# Plots medians with confidence intervals

fullData = read.csv("../../../results/tradeoff/pcfg/listener-curve-ci-median.tsv", sep="\t")

languages = read.csv("../../corpus_size/corpusSizes.tsv", sep="\t")
languages = languages$Language

for(language in languages) {
    dataL = read.csv(paste("../../../results/raw/pcfg/",language,"_pcfg_decay_after_tuning.tsv", sep=""), sep="\t")
    UnigramCE = mean(dataL$UnigramCE)
    data = fullData %>% filter(Language == language) %>% mutate(Type = factor(Type, levels = c("RANDOM_BY_TYPE", "GROUND")))
    plot = ggplot(data, aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianEmpirical), fill=Type, color=Type))
    plot = plot + geom_line(size=2)
    plot = plot + theme_classic() 
    plot = plot + theme(legend.position="none") 
    plot = plot + geom_line(data=data %>% filter(Type == "RANDOM_BY_TYPE"), aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianLower)), linetype="dashed")
    plot = plot + geom_line(data=data %>% filter(Type == "RANDOM_BY_TYPE"), aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianUpper)), linetype="dashed")
#    plot = plot + xlab("Memory") + ylab("Median Surprisal")
    plot = plot + theme(axis.text = element_text(size=50), axis.title=element_blank())
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIANS_onlyWordForms_boundedVocab-pcfg.pdf", sep=""))
}



