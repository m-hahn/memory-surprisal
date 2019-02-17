


memSurpPlot_onlyWordForms_characters = function(language) {
    data = read.csv(paste(language,"_after_tuning_onlyWordForms_characters.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL"))
    data = data %>% rename(Entropy_Rate=Residual)
    plot = ggplot(data, aes(x=Memory, y=Entropy_Rate, group=Type, fill=Type, color=Type)) +  geom_point() + theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-entropy-memory_onlyWordForms_characters.pdf", sep=""))
    return(plot)
}

plot = memSurpPlot_onlyWordForms_characters("Polish-LFG")
plot = memSurpPlot_onlyWordForms_characters("Russian")
plot = memSurpPlot_onlyWordForms_characters("Korean")





memListenerSurpPlot_onlyWordForms_characters = function(language) {
    data = read.csv(paste(language,"_decay_after_tuning_onlyWordForms_characters.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL"))


data2 = data %>% filter(Distance==1) %>% mutate(ConditionalMI=0, Distance=0)
data = rbind(data2, data)

    data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI), Surprisal=UnigramCE-CumulativeMI)

    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory_onlyWordForms_characters.pdf", sep=""))
    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory-by-run_onlyWordForms_characters.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory-by-run_onlyWordForms_characters.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory_onlyWordForms_characters.pdf", sep=""))



    return(plot)
}

plot = memListenerSurpPlot_onlyWordForms_characters("Russian")
plot = memListenerSurpPlot_onlyWordForms_characters("Polish-LFG")
plot = memListenerSurpPlot_onlyWordForms_characters("Korean")



