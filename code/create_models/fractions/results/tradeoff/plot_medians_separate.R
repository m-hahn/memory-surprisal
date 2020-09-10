library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

# Plots medians with confidence intervals

language="Russian"

for(fraction in c(500, 2000)) {

  fullData = read.csv(paste(language, "_", "listener-curve-ci-median_", fraction, ".tsv", sep=""), sep="\t")
  
   
  GREEN = "#009E73"
  BLUE  = "#0072B2"
  RED   = "#D55E00"

  dataL = read.csv(paste("~/CS_SCR/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_fraction", fraction, ".tsv", sep=""), sep="\t") %>% filter(Language == language)

  data=fullData
  data$UnigramCE = mean(dataL$UnigramCE)
 plot = ggplot(data, aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianEmpirical), fill=Type, color=Type)) +
	  geom_line(size=2) +
	  theme_classic() +
#	  theme(legend.position="none") +
	  geom_line(aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianLower)), linetype="dashed") +
	  geom_line(aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianUpper)), linetype="dashed") +
	  theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=30), legend.position="none")
  ggsave(plot, file=paste("figures/",language,"_", fraction, "-listener-surprisal-memory-MEDIANS_onlyWordForms_boundedVocab.pdf", sep=""), height=20, width=20)

}
