library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

# Plots medians with confidence intervals


  RED = "#F8766D"
  GREEN = "#7CAE00"
  BLUE = "#00BFC4"
  PURPLE = "#C77CFF"
  SCALE = c( GREEN, RED, PURPLE, BLUE)


fullData = read.csv("../../results/tradeoff/listener-curve-ci-median_infostruc.tsv", sep="\t")

languages=c("Czech-PDT")

for(language in languages) {
    dataL = read.csv(paste("../../raw/neural/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_infostruc.tsv", sep=""), sep="\t")
    UnigramCE = mean(dataL$UnigramCE)
    data = fullData %>% filter(Language == language)
    plot = ggplot(data, aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianEmpirical), fill=Type, color=Type)) + 
	    geom_line(size=2) + 
	    theme_classic() + 
	    theme(legend.position="none") + 
	    geom_line(aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianLower)), linetype="dashed") + 
	    geom_line(aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianUpper)), linetype="dashed") +  
	    theme(axis.title=element_text(size=20), axis.text = element_text(size=20))
    plot = plot + xlab("Memory (bits)") + ylab("Surprisal (bits)") + scale_colour_manual(values=SCALE) + scale_fill_manual(values=SCALE)
 #   plot = plot + theme(text = element_text(size=30))
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIANS_onlyWordForms_boundedVocab.pdf", sep=""), height=4, width=4)
}

