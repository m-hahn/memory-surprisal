library(tidyr)
library(dplyr)
library(ggplot2)

######################################

data = read.csv("results.tsv", sep="\t")


data = data %>% filter(Script == "forWords_Sesotho_RandomOrder_FormsWordsGraphemes_HeldoutClip.py")

unigramCE = mean(data$UnigramCE)

data$Type = ifelse(data$Model %in% c("REAL", "RANDOM", "REVERSE"), as.character(data$Model), "Optimized")

data_ = data %>% group_by(Distance, Type) %>% summarise(MI=median(MI))
data_ = data_ %>% filter(Distance <= 7)
data_ = data_ %>% filter(Type %in% c("REAL", "RANDOM"))
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
ggsave(plot, file=paste("figures/Sesotho-suffixes-byPhonemes-it-heldout.pdf", sep=""), height=4, width=4)




data = read.csv("results_interpolated.tsv", sep="\t")


data = data %>% filter(Script == "forWords_Sesotho_RandomOrder_FormsWordsGraphemes_HeldoutClip.py")

data = data %>% group_by(Type, Memory) %>% summarise(Surprisal=unigramCE-median(MI))

data_ = data %>% filter(Type %in% c("REAL", "RANDOM"))

plot = ggplot(data_, aes(x=Memory, y=Surprisal, color=Type)) + geom_line() #size=2)
plot = plot + theme_bw()
plot = plot + theme(axis.title.x=element_text(size=20), axis.title.y=element_text(size=20), axis.text = element_text(size=20))
plot = plot + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position="none")
plot = plot + theme(axis.line = element_line(colour = "black"),
            panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                    panel.background = element_blank())
ggsave(plot, file=paste("figures/Sesotho-suffixes-byPhonemes-memsurp-heldout.pdf", sep=""), height=4, width=4)


######################################

data = read.csv("results_auc.tsv", sep="\t")

data = data %>% filter(Script == "forWords_Sesotho_RandomOrder_FormsWordsGraphemes_HeldoutClip.py")


data$Type = ifelse(data$Model %in% c("REAL", "RANDOM", "REVERSE"), as.character(data$Model), "Optimized")

data_ = data %>% filter(Type %in% c("REAL", "RANDOM", "Optimized"))


plot = ggplot(data_, aes(x=AUC, y=1, color=Type, fill=Type)) + geom_bar(stat="identity") #size=2)
plot = plot + theme_bw()
plot = plot + theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=20))
plot = plot + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position="none")
plot = plot + theme(axis.line = element_line(colour = "black"),
            panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                    panel.background = element_blank())
ggsave(plot, file=paste("figures/Sesotho-suffixes-byPhonemes-auc-heldout.pdf", sep=""), height=4, width=4)

barWidth = (max(data$AUC) - min(data$AUC))/30

plot = ggplot(data_, aes(x=AUC, fill=Type, color=Type))
plot = plot + theme_classic()
plot = plot + theme(legend.position="none")
plot = plot + geom_density(data= data_%>%filter(Type == "RANDOM"), aes(y=..scaled..)) 
plot = plot + geom_bar(data = data_ %>% filter(!(Type %in% c("RANDOM"))) %>% group_by(Type) %>% summarise(AUC=mean(AUC)) %>% mutate(y=1),  aes(y=y, group=Type), width=barWidth, stat="identity", position = position_dodge())
ggsave(plot, file=paste("figures/Sesotho-suffixes-byPhonemes-auc-hist-heldout.pdf", sep=""), height=4, width=4)



