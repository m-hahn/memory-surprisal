library(tidyr)
library(dplyr)
library(ggplot2)

# Plots median difference with confidence intervals

fullData = read.csv("../../results/tradeoff/listener-curve-ci-median_diff.tsv", sep="\t")

memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    data = fullData %>% filter(Language == language)
    plot = ggplot(data, aes(x=Memory, y=-EmpiricalMedianDiff, fill=Type, color=Type)) + geom_line(size=2)+ theme_classic() + theme(legend.position="none") + geom_line(aes(x=Memory, y=-MedianDiff_Lower), linetype="dashed") + geom_line(aes(x=Memory, y=-MedianDiff_Upper), linetype="dashed")
    plot = plot + xlab("Memory") + ylab("Difference in Median Surprisal")
    plot = plot + theme(text = element_text(size=30))
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIAN_DIFFS_onlyWordForms_boundedVocab.pdf", sep=""))
    return(plot)
}

languages = read.csv("languages.tsv", sep="\t")

for(language in languages) {
   memListenerSurpPlot_onlyWordForms_boundedVocab(language)
}

