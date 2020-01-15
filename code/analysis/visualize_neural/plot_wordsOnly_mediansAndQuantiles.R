library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

# Plots medians with confidence intervals

fullData = read.csv("../../results/tradeoff/listener-curve-ci-median.tsv", sep="\t") 

randomRuns = read.csv("../../results/tradeoff/listener-curve-interpolated.tsv", sep="\t") %>% filter(Type == "RANDOM_BY_TYPE")

library(MASS)

memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    dataL = read.csv(paste("../../results/raw/word-level/",language,"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", sep=""), sep="\t")
    UnigramCE = mean(dataL$UnigramCE)
    dataR = randomRuns %>% filter(Language == language)
    data = fullData %>% filter(Language == language)
    plot = ggplot(data, aes(x=Memory, y=UnigramCE-MedianEmpirical, fill=Type, color=Type)) 
    plot = plot + geom_line(size=2)
    plot = plot + theme_classic() + theme(legend.position="none") 
    plot = plot + geom_line(aes(x=Memory, y=UnigramCE-MedianLower), linetype="dashed") + geom_line(aes(x=Memory, y=UnigramCE-MedianUpper), linetype="dashed")

    quantiles = data.frame(Memory = c(), MI = c(), Density = c())
    for(i in (1:max(dataR$Position))) {
       dataRP = dataR  %>% filter(Position == i )
       quants = quantile(dataRP$Surprisal, (1:9)/10)
       dens = data.frame(MI=quants, Quantile=(1:9)/10)
       quantiles = rbind(quantiles, dens %>% mutate(Memory = dataRP$Memory[[1]]))
    }
    quantiles$Type = "RANDOM_BY_TYPE"
    
    plot = plot + geom_line(data = quantiles, aes(x=Memory, y=UnigramCE-MI, group=Quantile), linetype = "dotted", size=0.5) 

    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIANS_QUANTILES_onlyWordForms_boundedVocab.pdf", sep=""))
    return(plot)
}

languages = read.csv("languages.tsv", sep="\t")

for(language in languages) {
   memListenerSurpPlot_onlyWordForms_boundedVocab(language)
}

