library(tidyr)
library(dplyr)
library(ggplot2)


RED = "#F8766D"
GREEN = "#7CAE00"
BLUE = "#00BFC4"
PURPLE = "#C77CFF"
SCALE = c( GREEN, RED, BLUE, PURPLE)
######################################

dataSesotho = read.csv("results_auc.tsv", sep="\t")
dataSesotho = dataSesotho %>% filter(Script == "forWords_Sesotho_RandomOrder_Normalized_HeldoutClip.py") %>% mutate(Language="Sesotho")


dataJapanese = read.csv("../../../Japanese/tradeoffs/analyze/results_auc.tsv", sep="\t")
dataJapanese = dataJapanese %>% filter(Script == "forWords_Japanese_RandomOrder_Normalized_FullData_Heldout.py") %>% mutate(Language="Japanese")

#dataJapanese$barWidth = (max(dataJapanese$AUC) - min(dataJapanese$AUC))/5
#dataSesotho$barWidth =(max(dataSesotho$AUC) - min(dataSesotho$AUC))/5


data = rbind(dataJapanese, dataSesotho)

data$Type = ifelse(data$Model %in% c("REAL", "RANDOM", "REVERSE"), as.character(data$Model), "Optimized")
data$Type = ifelse(data$Type %in% c("REAL"), "Real", as.character(data$Type))
data$Type = ifelse(data$Type %in% c("RANDOM"), "Random", as.character(data$Type))
data$Type = ifelse(data$Type %in% c("REVERSE"), "Reverse", as.character(data$Type))

data_ = data %>% filter(Type %in% c("Real", "Random", "Optimized", "Reverse"))


plot = ggplot(data_, aes(x=AUC, fill=Type, color=Type))
plot = plot + theme_classic()
plot = plot + xlab("Area under Curve") + ylab("Density")
plot = plot + theme(text=element_text(size=30))
plot = plot + geom_density(data= data_%>%filter(Type == "Random"), aes(y=..scaled..)) 
plot = plot + geom_bar(data = data_ %>% filter(Language=="Japanese", !(Type %in% c("Random"))) %>% group_by(Language, Type) %>% summarise(AUC=mean(AUC), barWidth=0.1) %>% mutate(y=1),  aes(y=y, group=Type), width=0.02, stat="identity", position = position_dodge())
plot = plot + geom_bar(data = data_ %>% filter(Language=="Sesotho", !(Type %in% c("Random"))) %>% group_by(Language, Type) %>% summarise(AUC=mean(AUC), barWidth=0.1) %>% mutate(y=1),  aes(y=y, group=Type), width=0.01, stat="identity", position = position_dodge())
plot = plot + facet_wrap(~Language, scales = "free")
#ggsave(plot, file=paste("figures/Both-suffixes-byMorphemes-auc-hist-heldout.pdf", sep=""), height=4, width=12)

# Version with SD
plot = ggplot(data_, aes(x=AUC, fill=Type, color=Type))
plot = plot + theme_classic()
plot = plot + xlab("Area under Curve") + ylab("Density")
plot = plot + theme(text=element_text(size=30))
plot = plot + geom_density(data= data_%>%filter(Type == "Random"), aes(y=..scaled..)) 
plot = plot + geom_bar(data = data_ %>% filter(Language=="Japanese", !(Type %in% c("Random"))) %>% group_by(Language, Type) %>% summarise(AUC=mean(AUC), barWidth=0.08) %>% mutate(y=1),  aes(y=y, group=Type), width=0.02, stat="identity", position = position_dodge())
plot = plot + geom_errorbarh(data = data_ %>% filter(Language=="Japanese", (Type %in% c("Optimized"))) %>% group_by(Language, Type) %>% summarise(AUC_min=min(AUC), AUC_max=max(AUC), AUC_sd=sd(AUC), AUC=mean(AUC), barWidth=0.1) %>% mutate(y=0.5), aes(y=y, xmin=AUC_min, xmax=AUC_max))
plot = plot + geom_bar(data = data_ %>% filter(Language=="Sesotho", !(Type %in% c("Random"))) %>% group_by(Language, Type) %>% summarise(AUC=mean(AUC), barWidth=0.08) %>% mutate(y=1),  aes(y=y, group=Type), width=0.01, stat="identity", position = position_dodge())
plot = plot + geom_errorbarh(data = data_ %>% filter(Language=="Sesotho", (Type %in% c("Optimized"))) %>% group_by(Language, Type) %>% summarise(AUC_min=min(AUC), AUC_max=max(AUC), AUC_sd=sd(AUC), AUC=mean(AUC), barWidth=0.1) %>% mutate(y=0.5), aes(y=y, xmin=AUC_min, xmax=AUC_max))
plot = plot + facet_wrap(~Language, scales = "free")
plot = plot + scale_colour_manual(values=SCALE) + scale_fill_manual(values=SCALE)
ggsave(plot, file=paste("../figures/Both-suffixes-byMorphemes-auc-hist-heldout.pdf", sep=""), height=4, width=12)







1-mean(((data_ %>% filter(Language=="Japanese", (Type %in% c("Random"))))$AUC) < max(((data_ %>% filter(Language=="Japanese", (Type %in% c("Optimized"))))$AUC)))
1-mean(((data_ %>% filter(Language=="Japanese", (Type %in% c("Random"))))$AUC) < max(((data_ %>% filter(Language=="Japanese", (Type %in% c("Real"))))$AUC)))
1-mean(((data_ %>% filter(Language=="Sesotho", (Type %in% c("Random"))))$AUC) < max(((data_ %>% filter(Language=="Sesotho", (Type %in% c("Optimized"))))$AUC)))
1-mean(((data_ %>% filter(Language=="Sesotho", (Type %in% c("Random"))))$AUC) < max(((data_ %>% filter(Language=="Sesotho", (Type %in% c("Real"))))$AUC)))

# Variation in AUC for Optimized Grammars
sd(((data_ %>% filter(Language=="Japanese", (Type %in% c("Optimized"))))$AUC))


# Variation in AUC for Optimized Grammars
sd(((data_ %>% filter(Language=="Sesotho", (Type %in% c("Optimized"))))$AUC))



