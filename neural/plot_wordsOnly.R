


memSurpPlot_wordsOnly = function(language) {
    data = read.csv(paste(language,"_after_tuning_wordsOnly.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL"))
    data = data %>% rename(Entropy_Rate=Residual)
    data = data %>% filter(Entropy_Rate < 20)

    plot = ggplot(data, aes(x=Memory, y=Entropy_Rate, group=Type, fill=Type, color=Type)) +  geom_point() + theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-entropy-memory_wordsOnly.pdf", sep=""))
    return(plot)
}

plot = memSurpPlot_wordsOnly("LDC2012T05")
plot = memSurpPlot_wordsOnly("Korean")


memListenerSurpPlot_wordsOnly = function(language) {
    data = read.csv(paste(language,"_decay_after_tuning_wordsOnly.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL"))


data2 = data %>% filter(Distance==1) %>% mutate(ConditionalMI=0, Distance=0)
data = rbind(data2, data)

    data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI), Surprisal=UnigramCE-CumulativeMI)

    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory_wordsOnly.pdf", sep=""))
    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory-by-run_wordsOnly.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory-by-run_wordsOnly.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory_wordsOnly.pdf", sep=""))



    return(plot)
}

plot = memListenerSurpPlot_wordsOnly("LDC2012T05")



plot = memListenerSurpPlot_wordsOnly("Korean")




memSurpPlot_onlyWordForms_boundedVocab = function(language) {
    data = read.csv(paste(language,"_after_tuning_onlyWordForms_boundedVocab.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL"))
    data = data %>% rename(Entropy_Rate=Residual)
    plot = ggplot(data, aes(x=Memory, y=Entropy_Rate, group=Type, fill=Type, color=Type)) +  geom_point() + theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-entropy-memory_onlyWordForms_boundedVocab.pdf", sep=""))
    return(plot)
}

plot = memSurpPlot_onlyWordForms_boundedVocab("Russian")
plot = memSurpPlot_onlyWordForms_boundedVocab("Erzya-Adap")
plot = memSurpPlot_onlyWordForms_boundedVocab("Bambara-Adap")
plot = memSurpPlot_onlyWordForms_boundedVocab("Polish-LFG")
plot = memSurpPlot_onlyWordForms_boundedVocab("North_Sami")




memSurpPlot_onlyWordForms = function(language) {
    data = read.csv(paste(language,"_after_tuning_onlyWordForms.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL"))
    data = data %>% rename(Entropy_Rate=Residual)
    plot = ggplot(data, aes(x=Memory, y=Entropy_Rate, group=Type, fill=Type, color=Type)) +  geom_point() + theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-entropy-memory_onlyWordForms.pdf", sep=""))
    return(plot)
}

plot = memSurpPlot_onlyWordForms("German")
plot = memSurpPlot_onlyWordForms("TuebaJS")




memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    data = read.csv(paste(language,"_decay_after_tuning_onlyWordForms_boundedVocab.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL"))


data2 = data %>% filter(Distance==1) %>% mutate(ConditionalMI=0, Distance=0)
data = rbind(data2, data)

    data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI), Surprisal=UnigramCE-CumulativeMI)

    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory_onlyWordForms_boundedVocab.pdf", sep=""))
    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory-by-run_onlyWordForms_boundedVocab.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory-by-run_onlyWordForms_boundedVocab.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory_onlyWordForms_boundedVocab.pdf", sep=""))



    return(plot)
}

plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Arabic")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Catalan")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Czech")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Dutch")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Finnish")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("French")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("German")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Hindi")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Norwegian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Spanish")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Basque")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Bulgarian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Croatian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Estonian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Hebrew")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Japanese")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Polish")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Romanian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Slovak")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Slovenian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Swedish")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Afrikaans")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Chinese")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Danish")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Greek")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Hungarian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("North_Sami")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Persian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Serbian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Turkish")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Ukrainian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Vietnamese")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Amharic-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Armenian-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Breton-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Buryat-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Cantonese-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Faroese-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Kazakh-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Kurmanji-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Naija-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Thai-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Uyghur-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Bambara-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Erzya-Adap")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Maltese")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Latvian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Indonesian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Urdu")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Portuguese")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("English")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Italian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Russian")



memListenerSurpPlot_onlyWordForms = function(language) {
    data = read.csv(paste(language,"_decay_after_tuning_onlyWordForms.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL"))


data2 = data %>% filter(Distance==1) %>% mutate(ConditionalMI=0, Distance=0)
data = rbind(data2, data)

    data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI), Surprisal=UnigramCE-CumulativeMI)

    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory_onlyWordForms.pdf", sep=""))
    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory-by-run_onlyWordForms.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory-by-run_onlyWordForms.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory_onlyWordForms.pdf", sep=""))



    return(plot)
}

plot = memListenerSurpPlot_onlyWordForms("German")
plot = memListenerSurpPlot_onlyWordForms("TuebaJS")


