

fullData = read.csv("../results/tradeoff/listener-curve-histogram_byMem.tsv", sep="\t")

    library(tidyr)
    library(dplyr)
    library(ggplot2)


    transform = fullData %>% filter(Type == "RANDOM_BY_TYPE") %>% group_by(Language) %>% summarise(mean=mean(MI), sd=sd(MI), counts=NROW(MI))
    data = merge(fullData, transform, by=c("Language"))
    data = data %>% mutate(MI_z = (MI-mean)/sd)
    barWidth = 0.05 #(max(data$MI) - min(data$MI))/30
    plot = ggplot(data, aes(x=-MI_z, fill=Type, color=Type)) 
    plot = plot + theme_classic() 
#    plot = plot + theme(legend.position="none")   
    plot = plot + geom_density(data= data%>%filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled.., weight=1/counts))      
    plot = plot + geom_bar(data = data %>% filter(Type %in% c("REAL_REAL")) %>% group_by(Language, Type) %>% summarise(MI_z=mean(MI_z)) %>% mutate(y=1),  aes(y=y, group=Type), width=barWidth, stat="identity", position = position_dodge()) 
    ggsave(plot, file=paste("figures/full-REAL-listener-surprisal-memory-HIST_byMem_onlyWordForms_boundedVocab.pdf", sep=""))

    plot = ggplot(data, aes(x=-MI_z, fill=Type, color=Type)) 
    plot = plot + theme_classic() 
#    plot = plot + theme(legend.position="none")   
    plot = plot + geom_density(data= data%>%filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled.., weight=1/counts))      
    plot = plot + geom_bar(data = data %>% filter(Type %in% c("GROUND")) %>% group_by(Language, Type) %>% summarise(MI_z=mean(MI_z)) %>% mutate(y=1),  aes(y=y, group=Type), width=barWidth, stat="identity", position = position_dodge()) 
    ggsave(plot, file=paste("figures/full-GROUND-listener-surprisal-memory-HIST_byMem_onlyWordForms_boundedVocab.pdf", sep=""))


branching = read.csv("../code/branching_entropy_coarse.tsv", sep="\t")
data = merge(data, branching, by=c("Language"))

dataReal = data %>% filter(Type %in% c("REAL_REAL")) %>% group_by(Language, Type) %>% summarise(MI_z=mean(MI_z), BranchingEntropy=mean(BranchingEntropy)) %>% mutate(y=1)

    plot = ggplot(data, aes(x=-MI_z)) 
    plot = plot + theme_classic() 
#    plot = plot + theme(legend.position="none")   
    plot = plot + geom_density(data= data%>%filter(Type == "RANDOM_BY_TYPE"), aes(y=..scaled.., weight=1/counts))      
    plot = plot + geom_bar(data = dataReal,  aes(x=-MI_z, y=y, fill=BranchingEntropy), width=barWidth, stat="identity", position = position_dodge()) 


cor.test(dataReal$MI_z, dataReal$BranchingEntropy, method="spearman")

plot(dataReal$MI_z, dataReal$BranchingEntropy)


