

memSurpNgramsPlot = function(language) {
    data = read.csv(paste(language,"_ngrams_after_tuning.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL", "GROUND"))
    data = data %>% rename(Entropy_Rate=Residual)
    plot = ggplot(data, aes(x=Memory, y=Entropy_Rate, group=Type, fill=Type, color=Type)) +  geom_point() + theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-ngrams-entropy-memory.pdf", sep=""))
    return(plot)
}


plot = memSurpNgramsPlot("BKTreebank_Vietnamese")


plot = memSurpNgramsPlot("Estonian")
plot = memSurpNgramsPlot("Vietnamese")
plot = memSurpNgramsPlot("Slovenian")
plot = memSurpNgramsPlot("Finnish")
plot = memSurpNgramsPlot("Polish")
plot = memSurpNgramsPlot("Czech")
plot = memSurpNgramsPlot("Swedish")
plot = memSurpNgramsPlot("Hindi")
plot = memSurpNgramsPlot("Basque")
plot = memSurpNgramsPlot("German")
plot = memSurpNgramsPlot("Arabic")
plot = memSurpNgramsPlot("Turkish")
plot = memSurpNgramsPlot("Spanish")
plot = memSurpNgramsPlot("Latvian")
plot = memSurpNgramsPlot("Indonesian")
plot = memSurpNgramsPlot("Hungarian")
plot = memSurpNgramsPlot("Persian")
plot = memSurpNgramsPlot("Croatian")
plot = memSurpNgramsPlot("French")
plot = memSurpNgramsPlot("Chinese")
plot = memSurpNgramsPlot("Serbian")
plot = memSurpNgramsPlot("Slovak")
plot = memSurpNgramsPlot("North_Sami")
plot = memSurpNgramsPlot("Irish")
plot = memSurpNgramsPlot("Lithuanian")
plot = memSurpNgramsPlot("Armenian")
plot = memSurpNgramsPlot("Ukrainian")
plot = memSurpNgramsPlot("Armenian-Adap")
plot = memSurpNgramsPlot("Greek")
plot = memSurpNgramsPlot("Uyghur-Adap")
plot = memSurpNgramsPlot("Breton-Adap")
plot = memSurpNgramsPlot("Thai-Adap")
plot = memSurpNgramsPlot("Tamil")
plot = memSurpNgramsPlot("Faroese-Adap")
plot = memSurpNgramsPlot("Buryat-Adap")
plot = memSurpNgramsPlot("Naija-Adap")
plot = memSurpNgramsPlot("Cantonese-Adap")
plot = memSurpNgramsPlot("Japanese")
plot = memSurpNgramsPlot("Hebrew")
plot = memSurpNgramsPlot("Marathi")
plot = memSurpNgramsPlot("Dutch")
plot = memSurpNgramsPlot("Kazakh-Adap")
plot = memSurpNgramsPlot("Amharic-Adap")
plot = memSurpNgramsPlot("Afrikaans")
plot = memSurpNgramsPlot("Bulgarian")
plot = memSurpNgramsPlot("Danish")
plot = memSurpNgramsPlot("Catalan")
plot = memSurpNgramsPlot("Belarusian")
plot = memSurpNgramsPlot("Norwegian")
plot = memSurpNgramsPlot("Romanian")
plot = memSurpNgramsPlot("Kurmanji-Adap")

############################################
plot = memSurpNgramsPlot("Bambara-Adap")
plot = memSurpNgramsPlot("Erzya-Adap")
plot = memSurpNgramsPlot("Maltese")


memListenerSurpNgramsPlot = function(language) {
    data = read.csv(paste(language,"_ngrams_decay_after_tuning.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL"))

#x = (2:20)
    # ggplot(data, aes(x=Distance, y=ConditionalMI, group=Type, color=Type)) + geom_line() + geom_line(data=data.frame(x=x, y=exp(-0.7*x)), aes(x=x, y=y, group=NULL), color="green") # THEORetically informed
#ggplot(data, aes(x=Distance, y=ConditionalMI, group=Type, color=Type)) + geom_line() + geom_line(data=data.frame(x=x, y=exp(-x)), aes(x=x, y=y, group=NULL), color="green")
# ggplot(data, aes(x=Distance, y=ConditionalMI, group=Type, color=Type)) + geom_line() + geom_line(data=data.frame(x=x, y=0.4/(x*log(x)^2)), aes(x=x, y=y, group=NULL), color="green")
data2 = data %>% filter(Distance==1) %>% mutate(ConditionalMI=0, Distance=0)
data = rbind(data2, data)


    data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI), Surprisal=UnigramCE-CumulativeMI)

    #plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    #ggsave(plot, file=paste(language,"-ngrams-listener-reverse-surprisal-memory.pdf", sep=""))
    #plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    #ggsave(plot, file=paste(language,"-ngrams-listener-reverse-surprisal-memory-by-run.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-ngrams-listener-surprisal-memory-by-run.pdf", sep=""))

    #plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    #ggsave(plot, file=paste(language,"-ngrams-listener-surprisal-memory.pdf", sep=""))



    return(plot)
}


plot = memListenerSurpNgramsPlot("BKTreebank_Vietnamese")


plot = memListenerSurpNgramsPlot("Estonian")
plot = memListenerSurpNgramsPlot("Vietnamese")
plot = memListenerSurpNgramsPlot("Slovenian")
plot = memListenerSurpNgramsPlot("Finnish")
plot = memListenerSurpNgramsPlot("Polish")
plot = memListenerSurpNgramsPlot("Czech")
plot = memListenerSurpNgramsPlot("Swedish")
plot = memListenerSurpNgramsPlot("Hindi")
plot = memListenerSurpNgramsPlot("Basque")
plot = memListenerSurpNgramsPlot("German")
plot = memListenerSurpNgramsPlot("Arabic")
plot = memListenerSurpNgramsPlot("Turkish")
plot = memListenerSurpNgramsPlot("Spanish")
plot = memListenerSurpNgramsPlot("Latvian")
plot = memListenerSurpNgramsPlot("Indonesian")
plot = memListenerSurpNgramsPlot("Hungarian")
plot = memListenerSurpNgramsPlot("Persian")
plot = memListenerSurpNgramsPlot("Croatian")
plot = memListenerSurpNgramsPlot("French")
plot = memListenerSurpNgramsPlot("Chinese")
plot = memListenerSurpNgramsPlot("Serbian")
plot = memListenerSurpNgramsPlot("Slovak")
plot = memListenerSurpNgramsPlot("North_Sami")
plot = memListenerSurpNgramsPlot("Irish")
plot = memListenerSurpNgramsPlot("Lithuanian")
plot = memListenerSurpNgramsPlot("Armenian")
plot = memListenerSurpNgramsPlot("Ukrainian")
plot = memListenerSurpNgramsPlot("Armenian-Adap")
plot = memListenerSurpNgramsPlot("Greek")
plot = memListenerSurpNgramsPlot("Uyghur-Adap")
plot = memListenerSurpNgramsPlot("Breton-Adap")
plot = memListenerSurpNgramsPlot("Thai-Adap")
plot = memListenerSurpNgramsPlot("Tamil")
plot = memListenerSurpNgramsPlot("Faroese-Adap")
plot = memListenerSurpNgramsPlot("Buryat-Adap")
plot = memListenerSurpNgramsPlot("Naija-Adap")
plot = memListenerSurpNgramsPlot("Cantonese-Adap")
plot = memListenerSurpNgramsPlot("Japanese")
plot = memListenerSurpNgramsPlot("Hebrew")
plot = memListenerSurpNgramsPlot("Marathi")
plot = memListenerSurpNgramsPlot("Dutch")
plot = memListenerSurpNgramsPlot("Kazakh-Adap")
plot = memListenerSurpNgramsPlot("Amharic-Adap")
plot = memListenerSurpNgramsPlot("Afrikaans")
plot = memListenerSurpNgramsPlot("Bulgarian")
plot = memListenerSurpNgramsPlot("Danish")
plot = memListenerSurpNgramsPlot("Catalan")
plot = memListenerSurpNgramsPlot("Belarusian")
plot = memListenerSurpNgramsPlot("Norwegian")
plot = memListenerSurpNgramsPlot("Romanian")
plot = memListenerSurpNgramsPlot("Kurmanji-Adap")
plot = memListenerSurpNgramsPlot("Bambara-Adap")
plot = memListenerSurpNgramsPlot("Erzya-Adap")
plot = memListenerSurpNgramsPlot("Maltese")









