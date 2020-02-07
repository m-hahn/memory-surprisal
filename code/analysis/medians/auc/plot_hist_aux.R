languages = read.csv("../../../ud_languages.txt", sep="\t", header=FALSE)$V1

data = read.csv("../../../../results/tradeoff/listener-curve-auc.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)


language = "Afrikaans"

for(language in languages) {
   d2 = data[data$Language == language,]
  
   ground = median(d2[d2$Type == "GROUND",]$AUC)
   real = median(d2[d2$Type == "REAL_REAL",]$AUC)
   random = d2[d2$Type == "RANDOM_BY_TYPE",]
    barWidth = (max(d2$AUC) - min(d2$AUC))/30

   plot = ggplot(d2, aes(x=AUC, fill=Type, color=Type))
   plot = plot + theme_classic()
   plot = plot + theme(legend.position="none")  
   plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled..))     
   plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("REAL_REAL", "GROUND")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type), width=barWidth, stat="identity", position = position_dodge()) 
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_AUC_onlyWordForms_boundedVocab.pdf", sep=""))

   plot = ggplot(d2, aes(x=AUC, fill=Type, color=Type))
   plot = plot + theme_classic()
   plot = plot + theme(legend.position="none")  
   plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled..))     
   plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("REAL_REAL")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type), width=barWidth, stat="identity", position = position_dodge()) 
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_AUC_onlyWordForms_boundedVocab_REAL.pdf", sep=""))
}
