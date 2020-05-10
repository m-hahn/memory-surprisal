languages = c("Czech-PDT")

data = read.csv("../../../results/tradeoff/listener-curve-auc.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)

for(language in languages) {
  d2 = data[data$Language == language,]
  
  real = median(d2[d2$Type == "REAL_REAL",]$AUC)
  ground = median(d2[d2$Type == "GROUND",]$AUC)
  random = d2[d2$Type == "RANDOM_BY_TYPE",]
  random_infostruc = d2[d2$Type == "RANDOM_INFOSTRUC",]
 
  cat(language,mean(random$AUC < real), binom.test(sum(random$AUC < real), nrow(random), alternative="two.sided")$p.value, mean(random_infostruc$AUC < real), binom.test(sum(random_infostruc$AUC < real), nrow(random), alternative="two.sided")$p.value, "\n", sep="\t")
}


