library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

# Plots medians with confidence intervals

fullData = read.csv("../../../results/tradeoff/listener-curve-ci-median.tsv", sep="\t")

languages = read.csv("languages.tsv")$Language

#GREEN = "#009E73"
#BLUE  = "#0072B2"
#RED   = "#D55E00"
#

RED = "#F8766D"
GREEN  = "#00BA38"
BLUE   = "#619CFF"


for(language in languages) {
  dataL = read.csv(paste("../../../results/raw/word-level/",language,"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", sep=""), sep="\t")
  
  dataL = dataL %>% group_by(Distance, Type) %>% summarise(ConditionalMI = median(ConditionalMI))
  
  dataL = dataL %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL", "GROUND"))
  
  plot = ggplot(dataL %>% filter(Distance < 5), aes(x=Distance, y=1.44*ConditionalMI, color=Type)) + geom_line(size=2)
  plot = plot + theme_bw()
  plot = plot + theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=30))
  plot = plot + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position="none") 
  plot = plot + theme(axis.line = element_line(colour = "black"),
	      panel.grid.major = element_blank(),
	          panel.grid.minor = element_blank(),
	          panel.border = element_blank(),
		      panel.background = element_blank())  + scale_colour_manual(values=c( GREEN, RED, BLUE))
  ggsave(plot, file=paste("figures/",language,"-it_mle.pdf", sep=""), height=4, width=4)
}
