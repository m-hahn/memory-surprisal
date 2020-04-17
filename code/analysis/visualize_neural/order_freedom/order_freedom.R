fullData = read.csv("../../../../results/tradeoff/listener-curve-histogram_byMem.tsv", sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    transform = fullData %>% filter(Type == "RANDOM_BY_TYPE") %>% group_by(Language) %>% summarise(mean=mean(MI), sd=sd(MI), counts=NROW(MI))
    data = merge(fullData, transform, by=c("Language"))
    data = data %>% mutate(MI_z = (MI-mean)/sd)
    barWidth = 0.05 #(max(data$MI) - min(data$MI))/30

branching = read.csv("../../../order-freedom/branching_entropy/branching_entropy.tsv", sep="\t")
data = merge(data, branching, by=c("Language"))

dataReal = data %>% filter(Type %in% c("REAL_REAL")) %>% group_by(Language, Type) %>% summarise(MI_z=mean(MI_z), MIDiff = mean(MI-mean), BranchingEntropy=mean(BranchingEntropy)) %>% mutate(y=1)
#cor.test(dataReal$MI_z, dataReal$BranchingEntropy, method="spearman")
cor.test(dataReal$MIDiff, dataReal$BranchingEntropy, method="spearman")
plot = ggplot(dataReal, aes(x=-MIDiff, y=BranchingEntropy)) + geom_point()
plot = plot + theme_classic() 
plot = plot + theme(legend.position="none")   
plot = plot + geom_text(aes(label=Language), hjust=0.8, vjust=1.1)
plot = plot + xlab("Surprisal Difference")
plot = plot + ylab("Branching Direction Entropy")
plot = plot + theme(axis.title=element_text(size=30))

ggsave(plot, file=paste("../figures/surprisal-branching-entropy-REAL.pdf", sep=""), width=15, height=7)



dataGround = data %>% filter(Type %in% c("GROUND")) %>% group_by(Language, Type) %>% summarise(MI_z=mean(MI_z), MIDiff = mean(MI-mean), BranchingEntropy=mean(BranchingEntropy)) %>% mutate(y=1)
#cor.test(dataGround$MI_z, dataGround$BranchingEntropy, method="spearman")
cor.test(dataGround$MIDiff, dataGround$BranchingEntropy, method="spearman")
plot = ggplot(dataGround, aes(x=-MIDiff, y=BranchingEntropy)) + geom_point()
plot = plot + theme_classic() 
plot = plot + theme(legend.position="none")   
plot = plot + geom_text(aes(label=Language), hjust=0.8, vjust=1.1)
#plot = plot + xlim(-2.5, 0.5)

ggsave(plot, file=paste("../figures/surprisal-branching-entropy-GROUND.pdf", sep=""))






dataReal = data %>% filter(Type %in% c("REAL_REAL")) %>% group_by(Language, Type) %>% summarise(MI_z=mean(MI_z), MIDiff = mean(MI-mean), BranchingEntropy=mean(BranchingEntropy)) %>% mutate(y=1)
#cor.test(dataReal$MI_z, dataReal$BranchingEntropy, method="spearman")
cor.test(dataReal$MIDiff, dataReal$BranchingEntropy, method="spearman")
plot = ggplot(dataReal, aes(x=-MIDiff, y=BranchingEntropy)) + geom_point()
plot = plot + theme_classic() 
plot = plot + theme(legend.position="none")   
plot = plot + geom_text(aes(label=Language), hjust=0.8, vjust=1.1)
plot = plot + xlab("Surprisal Difference")
plot = plot + ylab("Branching Direction Entropy")
plot = plot + theme(axis.title=element_text(size=30))
plot = plot + geom_segment(data=data.frame(x=-0.0999, xend=-0.23, y=0.328, yend=0.2), aes(x=x, xend=xend, y=y, yend=yend), size=2, arrow=arrow(), color="red")
ggsave(plot, file=paste("../figures/surprisal-branching-entropy-REAL-infostruc.pdf", sep=""), width=15, height=7)

#   Language       Type       MI_z MIDiff BranchingEntropy     y
#14 Czech          REAL_REAL 2.78  0.0999            0.328     1




