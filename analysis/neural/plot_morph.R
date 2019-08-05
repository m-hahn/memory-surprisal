

memSurpPlot = function(language) {
    data = read.csv(paste(language,"_after_tuning.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL", "GROUND"))
    data = data %>% rename(Entropy_Rate=Residual)
    plot = ggplot(data, aes(x=Memory, y=Entropy_Rate, group=Type, fill=Type, color=Type)) +  geom_point() + theme_classic() #+ theme(legend.position="none")
    ggsave(plot, file=paste(language,"-entropy-memory.pdf", sep=""))
    return(plot)
}


plot = memSurpPlot("BKTreebank_Vietnamese")


plot = memSurpPlot("Estonian")
plot = memSurpPlot("Vietnamese")
plot = memSurpPlot("Slovenian")
plot = memSurpPlot("Finnish")
plot = memSurpPlot("Polish")
plot = memSurpPlot("Czech")
plot = memSurpPlot("Swedish")
plot = memSurpPlot("Hindi")
plot = memSurpPlot("Basque")
plot = memSurpPlot("German")
plot = memSurpPlot("Arabic")
plot = memSurpPlot("Turkish")
plot = memSurpPlot("Spanish")
plot = memSurpPlot("Latvian")
plot = memSurpPlot("Indonesian")
plot = memSurpPlot("Hungarian")
plot = memSurpPlot("Persian")
plot = memSurpPlot("Croatian")
plot = memSurpPlot("French")
plot = memSurpPlot("Chinese")
plot = memSurpPlot("Serbian")
plot = memSurpPlot("Slovak")
plot = memSurpPlot("North_Sami")
plot = memSurpPlot("Irish")
plot = memSurpPlot("Lithuanian")
plot = memSurpPlot("Armenian")
plot = memSurpPlot("Ukrainian")
plot = memSurpPlot("Armenian-Adap")
plot = memSurpPlot("Greek")
plot = memSurpPlot("Uyghur-Adap")
plot = memSurpPlot("Breton-Adap")
plot = memSurpPlot("Thai-Adap")
plot = memSurpPlot("Tamil")
plot = memSurpPlot("Faroese-Adap")
plot = memSurpPlot("Buryat-Adap")
plot = memSurpPlot("Naija-Adap")
plot = memSurpPlot("Cantonese-Adap")
plot = memSurpPlot("Japanese")
plot = memSurpPlot("Hebrew")
plot = memSurpPlot("Marathi")
plot = memSurpPlot("Dutch")
plot = memSurpPlot("Kazakh-Adap")
plot = memSurpPlot("Amharic-Adap")
plot = memSurpPlot("Afrikaans")
plot = memSurpPlot("Bulgarian")
plot = memSurpPlot("Danish")
plot = memSurpPlot("Catalan")
plot = memSurpPlot("Belarusian")
plot = memSurpPlot("Norwegian")
plot = memSurpPlot("Romanian")
plot = memSurpPlot("Kurmanji-Adap")

############################################
plot = memSurpPlot("Bambara-Adap")
plot = memSurpPlot("Erzya-Adap")
plot = memSurpPlot("Maltese")


memListenerSurpPlot = function(language) {
    data = read.csv(paste(language,"_decay_after_tuning.tsv", sep=""), sep="\t")
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    data = data %>% filter(Type %in% c("RANDOM_BY_TYPE", "REAL_REAL", "GROUND"))

#x = (2:20)
    # ggplot(data, aes(x=Distance, y=ConditionalMI, group=Type, color=Type)) + geom_line() + geom_line(data=data.frame(x=x, y=exp(-0.7*x)), aes(x=x, y=y, group=NULL), color="green") # THEORetically informed
#ggplot(data, aes(x=Distance, y=ConditionalMI, group=Type, color=Type)) + geom_line() + geom_line(data=data.frame(x=x, y=exp(-x)), aes(x=x, y=y, group=NULL), color="green")
# ggplot(data, aes(x=Distance, y=ConditionalMI, group=Type, color=Type)) + geom_line() + geom_line(data=data.frame(x=x, y=0.4/(x*log(x)^2)), aes(x=x, y=y, group=NULL), color="green")
data2 = data %>% filter(Distance==1) %>% mutate(ConditionalMI=0, Distance=0)
data = rbind(data2, data)


    data = data %>% group_by(ModelID) %>% mutate(CumulativeMemory = cumsum(Distance*ConditionalMI), CumulativeMI = cumsum(ConditionalMI), Surprisal=UnigramCE-CumulativeMI)

    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory.pdf", sep=""))
    plot = ggplot(data, aes(x=Surprisal, y=CumulativeMemory, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-reverse-surprisal-memory-by-run.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=ModelID, fill=Type, color=Type, alpha=0.5)) + geom_line()+ theme_classic() + theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory-by-run.pdf", sep=""))

    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic()# + theme(legend.position="none")
    plot = ggplot(data, aes(x=CumulativeMemory, y=Surprisal, group=Type, fill=Type, color=Type, alpha=0.5)) + geom_smooth()+ theme_classic() #+ theme(legend.position="none")
    ggsave(plot, file=paste(language,"-listener-surprisal-memory.pdf", sep=""))



    return(plot)
}


plot = memListenerSurpPlot("BKTreebank_Vietnamese")


plot = memListenerSurpPlot("Estonian")
plot = memListenerSurpPlot("Vietnamese")
plot = memListenerSurpPlot("Slovenian")
plot = memListenerSurpPlot("Finnish")
plot = memListenerSurpPlot("Polish")
plot = memListenerSurpPlot("Czech")
plot = memListenerSurpPlot("Swedish")
plot = memListenerSurpPlot("Hindi")
plot = memListenerSurpPlot("Basque")
plot = memListenerSurpPlot("German")
plot = memListenerSurpPlot("Arabic")
plot = memListenerSurpPlot("Turkish")
plot = memListenerSurpPlot("Spanish")
plot = memListenerSurpPlot("Latvian")
plot = memListenerSurpPlot("Indonesian")
plot = memListenerSurpPlot("Hungarian")
plot = memListenerSurpPlot("Persian")
plot = memListenerSurpPlot("Croatian")
plot = memListenerSurpPlot("French")
plot = memListenerSurpPlot("Chinese")
plot = memListenerSurpPlot("Serbian")
plot = memListenerSurpPlot("Slovak")
plot = memListenerSurpPlot("North_Sami")
plot = memListenerSurpPlot("Irish")
plot = memListenerSurpPlot("Lithuanian")
plot = memListenerSurpPlot("Armenian")
plot = memListenerSurpPlot("Ukrainian")
plot = memListenerSurpPlot("Armenian-Adap")
plot = memListenerSurpPlot("Greek")
plot = memListenerSurpPlot("Uyghur-Adap")
plot = memListenerSurpPlot("Breton-Adap")
plot = memListenerSurpPlot("Thai-Adap")
plot = memListenerSurpPlot("Tamil")
plot = memListenerSurpPlot("Faroese-Adap")
plot = memListenerSurpPlot("Buryat-Adap")
plot = memListenerSurpPlot("Naija-Adap")
plot = memListenerSurpPlot("Cantonese-Adap")
plot = memListenerSurpPlot("Japanese")
plot = memListenerSurpPlot("Hebrew")
plot = memListenerSurpPlot("Marathi")
plot = memListenerSurpPlot("Dutch")
plot = memListenerSurpPlot("Kazakh-Adap")
plot = memListenerSurpPlot("Amharic-Adap")
plot = memListenerSurpPlot("Afrikaans")
plot = memListenerSurpPlot("Bulgarian")
plot = memListenerSurpPlot("Danish")
plot = memListenerSurpPlot("Catalan")
plot = memListenerSurpPlot("Belarusian")
plot = memListenerSurpPlot("Norwegian")
plot = memListenerSurpPlot("Romanian")
plot = memListenerSurpPlot("Kurmanji-Adap")
plot = memListenerSurpPlot("Bambara-Adap")
plot = memListenerSurpPlot("Erzya-Adap")
plot = memListenerSurpPlot("Maltese")













