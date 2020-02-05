languages = read.csv("../../../ud_languages.txt", sep="\t", header=FALSE)$V1

data = read.csv("../../../../results/tradeoff/listener-curve-auc.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)

sink("binomial-analysis.tsv")
cat("Language", "MeanLessThanReal", "pReal", "MeanLessThanGround", "pGround", "\n", sep="\t")
for(language in languages) {
  d2 = data[data$Language == language,]
  
  ground = median(d2[d2$Type == "GROUND",]$AUC)
  real = median(d2[d2$Type == "REAL_REAL",]$AUC)
  random = d2[d2$Type == "RANDOM_BY_TYPE",]
  
  cat(language,mean(random$AUC < real), binom.test(sum(random$AUC < real), nrow(random), alternative="two.sided")$p.value, mean(random$AUC < ground), binom.test(sum(random$AUC < ground), nrow(random), alternative="two.sided")$p.value, "\n", sep="\t")
}
sink()

