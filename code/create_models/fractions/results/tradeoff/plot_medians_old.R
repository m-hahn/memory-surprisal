library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

# Plots medians with confidence intervals

language="Czech"


fullData_10000 = read.csv(paste(language, "_", "listener-curve-ci-median_10000.tsv", sep=""), sep="\t") %>% mutate(Sentences = 10000)
fullData_1000 = read.csv(paste(language, "_", "listener-curve-ci-median_1000.tsv", sep=""), sep="\t") %>% mutate(Sentences = 1000)
fullData_20000 = read.csv(paste(language, "_", "listener-curve-ci-median_20000.tsv", sep=""), sep="\t") %>% mutate(Sentences = 20000)
fullData_2000 = read.csv(paste(language, "_", "listener-curve-ci-median_2000.tsv", sep=""), sep="\t") %>% mutate(Sentences = 2000)
fullData_5000 = read.csv(paste(language, "_", "listener-curve-ci-median_5000.tsv", sep=""), sep="\t") %>% mutate(Sentences = 5000)
fullData_500 = read.csv(paste(language, "_", "listener-curve-ci-median_500.tsv", sep=""), sep="\t") %>% mutate(Sentences = 500)



fullData = rbind(fullData_1000, fullData_1000, fullData_20000, fullData_2000, fullData_5000, fullData_500)


GREEN = "#009E73"
BLUE  = "#0072B2"
RED   = "#D55E00"

  dataL_500 = read.csv(paste("~/CS_SCR/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_fraction500.tsv", sep=""), sep="\t") %>% mutate(Sentences=500) %>% filter(Language == language)
  dataL_5000 = read.csv(paste("~/CS_SCR/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_fraction5000.tsv", sep=""), sep="\t") %>% mutate(Sentences=5000) %>% filter(Language == language)
  dataL_10000 = read.csv(paste("~/CS_SCR/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_fraction10000.tsv", sep=""), sep="\t") %>% mutate(Sentences=10000) %>% filter(Language == language)
  dataL_1000 = read.csv(paste("~/CS_SCR/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_fraction1000.tsv", sep=""), sep="\t") %>% mutate(Sentences=1000) %>% filter(Language == language)
  dataL_20000 = read.csv(paste("~/CS_SCR/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_fraction20000.tsv", sep=""), sep="\t") %>% mutate(Sentences=20000) %>% filter(Language == language)
  dataL_2000 = read.csv(paste("~/CS_SCR/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_fraction2000.tsv", sep=""), sep="\t") %>% mutate(Sentences=2000) %>% filter(Language == language)

  UnigramCE = rbind(dataL_500, dataL_5000, dataL_10000, dataL_1000, dataL_20000, dataL_2000) %>% group_by(Sentences) %>% summarise(UnigramCE=mean(UnigramCE))
  data = merge(fullData, UnigramCE, by=c("Sentences"))
 plot = ggplot(data, aes(x=Memory, y=UnigramCE-MedianEmpirical, fill=Type, color=Type)) +
	  geom_line(size=2) +
	  theme_classic() +
#	  theme(legend.position="none") +
	  geom_line(aes(x=Memory, y=UnigramCE-MedianLower), linetype="dashed") +
	  geom_line(aes(x=Memory, y=UnigramCE-MedianUpper), linetype="dashed") +
	  theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=30))+ # + scale_colour_manual(values=c( "#009E73", "#0072B2", "#D55E00"))
  facet_wrap(~Sentences, scales="free")
  ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIANS_onlyWordForms_boundedVocab.pdf", sep=""), height=20, width=20)

