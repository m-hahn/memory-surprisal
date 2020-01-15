library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

fullData_ConfidenceLowerBound = read.csv("../../../results/tradeoff/listener-curve-binomial-confidence-bound-quantile-noAssumption-Ngrams.tsv", sep="\t") #%>% filter(Type != "GROUND")
fullData_ConfidenceLowerBound_05 = read.csv("../../../results/tradeoff/listener-curve-binomial-confidence-bound-quantile-noAssumption-Ngrams-05.tsv", sep="\t") #%>% filter(Type != "GROUND")

fullData_BinomialTest = read.csv("../../../results/tradeoff/listener-curve-binomial-test-ngrams.tsv", sep="\t") #%>% filter(Type != "GROUND")

memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    data = fullData_ConfidenceLowerBound %>% filter(Language == language)
    data2 = fullData_BinomialTest %>% filter(Language == language)
    data3 = fullData_ConfidenceLowerBound_05 %>% filter(Language == language)
    #data = merge(data, data2, by=c("Language", "Position", "Type"))
    #data$Memory = data$Memory.x
    plot = ggplot(data, aes(x=Memory, y=LowerConfidenceBound, fill=Type, color=Type))
    plot = plot + geom_line(size=1, linetype="dotted") 
    plot = plot + geom_line(data=data3, size=2, linetype="dashed") 
    plot = plot + geom_line(data=data2, aes(x=Memory, y=BetterEmpirical), size=2)
    data2 = data2 %>% mutate(pValue_print = ifelse(round(pValue,5) == 0, "p<0.00001", paste("p=", round(pValue,5), sep="")))
    plot = plot + geom_text(data=data2 %>% filter(Position %% 9 == 0, Type == "REAL_REAL"), aes(x=Memory, y=BetterEmpirical+0.1, label=pValue_print), size=3)
    plot = plot + geom_text(data=data2 %>% filter(Position %% 9 == 0, Type == "GROUND"), aes(x=Memory, y=BetterEmpirical+0.05, label=pValue_print), size=3)
    plot = plot + theme_classic() 
    plot = plot + theme(legend.position="none") 
    plot = plot + ylim(0,1.1)
    plot = plot + ylab("Quantile")
    plot = plot + xlab("Memory")
    plot = plot + theme(text = element_text(size=20))
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-QUANTILES_onlyWordForms_boundedVocab_REAL_NGRAMS.pdf", sep=""), height=3.5, width=4.5)
    return(plot)
}

languages = read.csv("../../corpusSizes.tsv", sep="\t")
languages = languages$Language

for(language in languages) {
   memListenerSurpPlot_onlyWordForms_boundedVocab(language)
}

