library(ggplot2)
library(tidyr)
library(dplyr)

curves = read.csv("output/curves.tsv", sep=" ")
auc = read.csv("output/auc.tsv", sep=" ")


plot = ggplot(curves %>% filter(Script == "forWords_Celex_RandomOrder.py_model"), aes(x=Memory, y=Surprisal, color=Model, group=ModelID)) + geom_line()
ggsave(plot, file="figures/Japanese-suffixes.pdf")

plot = ggplot(curves %>% filter(Script == "forWords_Celex_RandomOrder_FormsWords.py_model"), aes(x=Memory, y=Surprisal, color=Model, group=ModelID)) + geom_line()
ggsave(plot, file="figures/Japanese-suffixes-words.pdf")


#ground = median(d2[d2$Type == "GROUND",]$AUC)
#real = median(d2[d2$Type == "REAL_REAL",]$AUC)
#random = d2[d2$Type == "RANDOM_BY_TYPE",]
#barWidth = (max(d2$AUC) - min(d2$AUC))/30
#
#plot = ggplot(d2, aes(x=AUC, fill=Type, color=Type))
#plot = plot + theme_classic()
#plot = plot + theme(legend.position="none")
#plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled..))
#plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("REAL_REAL", "GROUND")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type), width=barWidth, stat="identity", position = position_dodge())
#ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_AUC_onlyWordForms_boundedVocab.pdf", sep=""))
#
#
#
