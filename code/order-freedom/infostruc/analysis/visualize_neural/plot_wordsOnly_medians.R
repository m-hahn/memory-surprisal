library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

# Plots medians with confidence intervals

fullData = read.csv("../../results/tradeoff/listener-curve-ci-median_infostruc.tsv", sep="\t")

memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    dataL = read.csv(paste("../../raw/neural/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_infostruc.tsv", sep=""), sep="\t")
    UnigramCE = mean(dataL$UnigramCE)
    data = fullData %>% filter(Language == language)
    plot = ggplot(data, aes(x=Memory, y=UnigramCE-MedianEmpirical, fill=Type, color=Type)) + geom_line(size=2)+ theme_classic() + theme(legend.position="none") + geom_line(aes(x=Memory, y=UnigramCE-MedianLower), linetype="dashed") + geom_line(aes(x=Memory, y=UnigramCE-MedianUpper), linetype="dashed")    +  theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=20))
#    plot = plot + xlab("Memory") + ylab("Median Surprisal")
 #   plot = plot + theme(text = element_text(size=30))
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIANS_onlyWordForms_boundedVocab.pdf", sep=""), height=4, width=4)
    return(plot)
}

plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Czech-PDT")
