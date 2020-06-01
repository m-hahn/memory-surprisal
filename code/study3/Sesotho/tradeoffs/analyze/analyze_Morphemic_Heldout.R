library(tidyr)
library(dplyr)
library(ggplot2)

######################################

data = read.csv("results.tsv", sep="\t")
data = data %>% filter(Script == "forWords_Sesotho_RandomOrder_Normalized_HeldoutClip.py")

unigramCE = mean(data$UnigramCE)

data$Type = ifelse(data$Model %in% c("REAL", "RANDOM", "REVERSE"), as.character(data$Model), "Optimized")
data$Type = ifelse(data$Type %in% c("REAL"), "Real", as.character(data$Type))
data$Type = ifelse(data$Type %in% c("RANDOM"), "Random", as.character(data$Type))
data$Type = ifelse(data$Type %in% c("REVERSE"), "Reverse", as.character(data$Type))

data_ = data %>% group_by(Distance, Type) %>% summarise(MI=median(MI))
data_ = data_ %>% filter(Distance <= 7)
plot = ggplot(data_, aes(x=1+Distance, y=MI, color=Type)) + geom_line() #size=2)
plot = plot + theme_bw()
plot = plot + xlab("Distance") + ylab("Mutual Information")
plot = plot + theme(axis.title.x=element_text(size=20), axis.title.y=element_text(size=20), axis.text = element_text(size=20))
plot = plot + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position="none")
plot = plot + theme(axis.line = element_line(colour = "black"),
            panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                    panel.background = element_blank())
ggsave(plot, file=paste("../figures/Sesotho-suffixes-byMorphemes-it-heldout.pdf", sep=""), height=4, width=4)




data = read.csv("results_interpolated.tsv", sep="\t")


data = data %>% filter(Script == "forWords_Sesotho_RandomOrder_Normalized_HeldoutClip.py")

data = data %>% group_by(Type, Memory) %>% summarise(Surprisal=unigramCE-median(MI))

plot = ggplot(data, aes(x=Memory, y=Surprisal, color=Type)) + geom_line() #size=2)
plot = plot + theme_bw()
plot = plot + theme(axis.title.x=element_text(size=20), axis.title.y=element_text(size=20), axis.text = element_text(size=20))
plot = plot + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position="none")
plot = plot + theme(axis.line = element_line(colour = "black"),
            panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                    panel.background = element_blank())
ggsave(plot, file=paste("../figures/Sesotho-suffixes-byMorphemes-memsurp-heldout.pdf", sep=""), height=4, width=4)


######################################

data = read.csv("results_auc.tsv", sep="\t")
data = data %>% filter(Script == "forWords_Sesotho_RandomOrder_Normalized_HeldoutClip.py")


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
plot = plot + geom_bar(data = data_ %>% filter(!(Type %in% c("Random"))) %>% group_by(Type) %>% summarise(AUC=mean(AUC), barWidth=0.1) %>% mutate(y=1),  aes(y=y, group=Type), width=0.01, stat="identity", position = position_dodge())
ggsave(plot, file=paste("../figures/Sesotho-suffixes-byMorphemes-auc-hist-heldout.pdf", sep=""), height=4, width=6)



