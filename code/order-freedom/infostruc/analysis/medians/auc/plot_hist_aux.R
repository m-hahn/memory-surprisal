
data = read.csv("../../../results/tradeoff/listener-curve-auc.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)



languages=c("Czech-PDT")

for(language in languages) {
   d2 = data[data$Language == language,]
  
   real = median(d2[d2$Type == "REAL_REAL",]$AUC)
   random = d2[d2$Type == "RANDOM_BY_TYPE",]
   random_infostruc = d2[d2$Type == "RANDOM_INFOSTRUC",]
   barWidth = (max(d2$AUC) - min(d2$AUC))/30

#   plot = ggplot(d2, aes(x=AUC, fill=Type, color=Type))
#   plot = plot + theme_classic()
#   plot = plot + theme(legend.position="none")  
#   plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled..))     
#   plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("REAL_REAL")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type), width=barWidth, stat="identity", position = position_dodge()) 
#    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_AUC_onlyWordForms_boundedVocab_REAL.pdf", sep=""))

   plot = ggplot(d2, aes(x=AUC, color=Type))
   plot = plot + theme_classic()
   plot = plot + theme(legend.position="none")  
   plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_INFOSTRUC"), aes(y=..scaled..), size=2)     
   plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled..), size=2)     
   plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("REAL_REAL")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type, fill=Type, color=Type), width=barWidth, stat="identity", position = position_dodge()) 
   plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("GROUND")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type, fill=Type, color=Type), width=barWidth, stat="identity", position = position_dodge()) 
   plot = plot + ylab("")
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_AUC_onlyWordForms_boundedVocab_REAL-infostruc.pdf", sep=""))



   plot = ggplot(d2, aes(x=AUC, fill=Type, color=Type))
   plot = plot + theme_classic()
   plot = plot + theme(legend.position="none")  
   plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_INFOSTRUC"), aes(y=..scaled..), size=2)     
   plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("REAL_REAL")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type, fill=Type, color=Type), width=barWidth, stat="identity", position = position_dodge()) 
   plot = plot + ylab("")
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_AUC_onlyWordForms_boundedVocab_REAL-infostruc_REAL.pdf", sep=""))


   plot = ggplot(d2, aes(x=AUC, fill=Type, color=Type))
   plot = plot + theme_classic()
   plot = plot + theme(legend.position="none")  
   plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled..), size=2)     
   plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("GROUND")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type, fill=Type, color=Type), width=barWidth, stat="identity", position = position_dodge()) 
   plot = plot + ylab("")
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_AUC_onlyWordForms_boundedVocab_REAL-infostruc_GROUND.pdf", sep=""))





}
