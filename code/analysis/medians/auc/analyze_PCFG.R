languages = read.csv("../../../ud_languages.txt", sep="\t", header=FALSE)$V1

data = read.csv("../../../../results/tradeoff/listener-curve-auc-pcfg.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)

sink("binomial-analysis-pcfg.tsv")
cat("Language", "MeanLessThanGround", "pGround", "\n", sep="\t")
for(language in languages) {
  d2 = data[data$Language == language,]
  
  ground = median(d2[d2$Type == "GROUND",]$AUC)
#  real = median(d2[d2$Type == "REAL_REAL",]$AUC)
  random = d2[d2$Type == "RANDOM_BY_TYPE",]
  
  cat(language, mean(random$AUC < ground), binom.test(sum(random$AUC < ground), nrow(random), alternative="two.sided")$p.value, "\n", sep="\t")
}
sink()


data = read.csv("binomial-analysis-pcfg.tsv", sep="\t")

# Numerically, real orders are more efficient than <50% of baselines
data %>% filter(!(MeanLessThanGround < 0.5))

# Hochberg's step-up procedure
ground = (data %>% filter(MeanLessThanGround < 0.5))
ground = ground[order(ground$pGround),]
N = length(ground$pGround)
limit = 0.01/(N-(1:N)+1)
ground$rejectedNull = (ground$pGround <= limit)


