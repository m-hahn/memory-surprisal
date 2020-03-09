

library(tidyr)
library(dplyr)
library(ggplot2)


data = read.csv("output/perLanguageMeans.tsv", sep="\t")

data$Type = as.character(data$Script)

data$Type = "NONE"
data[data$Script == "optimizeDependencyLength_QuasiF.py",]$Type = "DepL"
data[data$Script == "optimizeGrammarForI1_10_Long.py",]$Type = "10"
data[data$Script == "optimizeGrammarForI1_7_Long.py",]$Type = "_7"
data[data$Script == "optimizeGrammarForI1_9_Long.py",]$Type = "_9"


plot = ggplot(data %>% filter(Type != "NONE"), aes(x=Type, y=Accuracy, fill=Type)) + geom_col() + facet_wrap(~Language)

ggsave("output/perLanguageMeans.pdf")

