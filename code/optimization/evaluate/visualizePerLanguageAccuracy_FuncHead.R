

library(tidyr)
library(dplyr)
library(ggplot2)


data = read.csv("output/perLanguageMeans_FuncHead.tsv", sep="\t")

data$Type = as.character(data$Script)

data[data$Script == "readDataDistCrossLGPUDepLengthMomentumEntropyUnbiasedBaseline_OrderBugFixed_NoPunct_NEWPYTORCH_AllCorpPerLang_BoundIterations_FuncHead_CoarseOnly.py",]$Type = "DepL"
data[data$Script == "readDataDistCrossGPUFreeAllTwoEqual_NoClip_ByCoarseOnly_FixObj_OnlyLangmod_Replication_Best.py",]$Type = "LSTM"


plot = ggplot(data, aes(x=Type, y=Accuracy, fill=Type)) + geom_col() + facet_wrap(~Language)

ggsave("output/perLanguageMeans_FuncHead.pdf")

