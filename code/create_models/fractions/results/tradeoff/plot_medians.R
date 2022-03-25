library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

# Plots medians with confidence intervals

language="Spanish"


fullData_2000 = read.csv(paste(language, "_", "listener-curve-ci-median_2000.tsv", sep=""), sep="\t") %>% mutate(Sentences = 2000)
fullData_500 = read.csv(paste(language, "_", "listener-curve-ci-median_500.tsv", sep=""), sep="\t") %>% mutate(Sentences = 500)



fullData = rbind(fullData_2000, fullData_500)


GREEN = "#009E73"
BLUE  = "#0072B2"
RED   = "#D55E00"

  dataL_500 = read.csv(paste("~/CS_SCR/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_fraction500.tsv", sep=""), sep="\t") %>% mutate(Sentences=500) %>% filter(Language == language)
  dataL_2000 = read.csv(paste("~/CS_SCR/",language,"_decay_after_tuning_onlyWordForms_boundedVocab_fraction2000.tsv", sep=""), sep="\t") %>% mutate(Sentences=2000) %>% filter(Language == language)

  UnigramCE = rbind(dataL_500, dataL_2000) %>% group_by(Sentences) %>% summarise(UnigramCE=mean(UnigramCE))
  data = merge(fullData, UnigramCE, by=c("Sentences"))
 plot = ggplot(data, aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianEmpirical), fill=Type, color=Type)) +
	  geom_line(size=2) +
	  theme_classic() +
#	  theme(legend.position="none") +
	  geom_line(aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianLower)), linetype="dashed") +
	  geom_line(aes(x=1.44*Memory, y=1.44*(UnigramCE-MedianUpper)), linetype="dashed") +
	  theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=30))+ # + scale_colour_manual(values=c( "#009E73", "#0072B2", "#D55E00"))
  facet_wrap(~Sentences, scales="free")
  ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIANS_onlyWordForms_boundedVocab.pdf", sep=""), height=20, width=20)


