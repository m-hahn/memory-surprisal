
data = read.csv("../../../results/tradeoff/listener-curve-auc.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)


  RED = "#F8766D"
  GREEN = "#7CAE00"
  BLUE = "#00BFC4"
  PURPLE = "#C77CFF"
  SCALE = c( GREEN, RED, PURPLE, BLUE)






languages=c("Czech-PDT")

for(language in languages) {
   d2 = data[data$Language == language,]
  
   real = median(d2[d2$Type == "REAL_REAL",]$AUC)
   random = d2[d2$Type == "RANDOM_BY_TYPE",]
   random_infostruc = d2[d2$Type == "RANDOM_INFOSTRUC",]
   barWidth = (max(d2$AUC) - min(d2$AUC))/30

   plot = ggplot(d2, aes(x=1.442695*1.442695*AUC, color=Type))
   plot = plot + theme_classic()
   plot = plot + theme(legend.position="none", axis.text=element_text(size=20), axis.title=element_text(size=20))
   plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_INFOSTRUC"), aes(y=..scaled..), size=2, color="#00BFC4")     
   plot = plot + geom_density(data= d2 %>% filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled..), size=2, color="#7CAE00")     
   plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("REAL_REAL")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type, fill=Type, color=Type), width=barWidth, stat="identity", position = position_dodge(), fill="#C77CFF", color="#C77CFF") 
   plot = plot + geom_bar(data = d2 %>% filter(Type %in% c("GROUND")) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type, fill=Type, color=Type), width=barWidth, stat="identity", position = position_dodge(), fill="#F8766D", color="#F8766D") 
   plot = plot + ylab("") + xlab("Area under Curve") + scale_colour_manual(values=SCALE) + scale_fill_manual(values=SCALE)
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-HIST_AUC_onlyWordForms_boundedVocab_REAL-infostruc.pdf", sep=""))
}

