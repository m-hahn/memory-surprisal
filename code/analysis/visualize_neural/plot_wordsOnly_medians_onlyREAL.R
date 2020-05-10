library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

# Plots medians with confidence intervals

fullData = read.csv("../../../results/tradeoff/listener-curve-ci-median.tsv", sep="\t") %>% filter(Type != "GROUND")


languages = read.csv("languages.tsv")$Language

for(language in languages) {
  dataL = read.csv(paste("../../../results/raw/word-level/",language,"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", sep=""), sep="\t")
  UnigramCE = mean(dataL$UnigramCE)
  data = fullData %>% filter(Language == language)
  plot = ggplot(data, aes(x=Memory, y=UnigramCE-MedianEmpirical, fill=Type, color=Type)) +
	  geom_line(size=2) +
	  theme_classic() +
	  theme(legend.position="none") +
	  geom_line(aes(x=Memory, y=UnigramCE-MedianLower), linetype="dashed") +
	  geom_line(aes(x=Memory, y=UnigramCE-MedianUpper), linetype="dashed") +
	  theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=30))
  ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIANS_onlyWordForms_boundedVocab_REAL.pdf", sep=""), height=4, width=4)
}

