data = read.csv("results.tsv", sep="\t")


library(tidyr)
library(dplyr)
library(ggplot2)

data = data %>% filter(Script == "forWords_Sesotho_RandomOrder_Normalized.py")


data$Type = ifelse(data$Model %in% c("REAL", "RANDOM"), as.character(data$Model), "Optimized")

plot = ggplot(data, aes(x=Distance, y=MI, color=Type)) + geom_line(size=2)
plot = plot + theme_bw()
plot = plot + theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=20))
plot = plot + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank(), legend.position="none")
plot = plot + theme(axis.line = element_line(colour = "black"),
            panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                    panel.background = element_blank())
ggsave(plot, file=paste("figures/Sesotho-suffixes-byMorphemes-it.pdf", sep=""), height=4, width=4)



plot = ggplot(data, aes(x=Memory, y=Surprisal, color=Type)) + geom_line() #size=2)
plot = plot + theme_bw()
plot = plot + theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=20))
plot = plot + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank()) #, legend.position="none")
plot = plot + theme(axis.line = element_line(colour = "black"),
            panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                    panel.background = element_blank())
ggsave(plot, file=paste("figures/Sesotho-suffixes-byMorphemes-memsurp.pdf", sep=""), height=4, width=4)

data = read.csv("results_auc.tsv", sep="\t")

data = data %>% filter(Script == "forWords_Sesotho_RandomOrder_Normalized.py")


data$Type = ifelse(data$Model %in% c("REAL", "RANDOM"), as.character(data$Model), "Optimized")

plot = ggplot(data, aes(x=AUC, y=1, color=Type, fill=Type)) + geom_bar(stat="identity") #size=2)
plot = plot + theme_bw()
plot = plot + theme(axis.title.x=element_blank(), axis.title.y=element_blank(), axis.text = element_text(size=20))
plot = plot + theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank()) #, legend.position="none")
plot = plot + theme(axis.line = element_line(colour = "black"),
            panel.grid.major = element_blank(),
                panel.grid.minor = element_blank(),
                panel.border = element_blank(),
                    panel.background = element_blank())
ggsave(plot, file=paste("figures/Sesotho-suffixes-byMorphemes-auc.pdf", sep=""), height=4, width=4)


