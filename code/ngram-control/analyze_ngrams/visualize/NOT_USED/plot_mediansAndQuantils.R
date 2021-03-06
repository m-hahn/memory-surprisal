# DOES not produce reasonable results. Better to just provide empirical quantile plots.  

library(tidyr)
library(dplyr)
library(ggplot2)

# Plots CIs for the quantile

# Plots medians with confidence intervals

fullData = read.csv("../../../results/tradeoff/ngrams/listener-curve-ci-median.tsv", sep="\t")
randomRuns = read.csv("../../../results/tradeoff/ngrams/listener-curve-interpolated.tsv", sep="\t") %>% filter(Type == "RANDOM_BY_TYPE")

library(MASS)

memListenerSurpPlot_onlyWordForms_boundedVocab = function(language) {
    library(tidyr)
    library(dplyr)
    library(ggplot2)
    dataL = read.csv(paste("../../../results/raw/ngrams/",language,"_ngrams_decay_after_tuning.tsv", sep=""), sep="\t")
    UnigramCE = mean(dataL$UnigramCE)

    dataR = randomRuns %>% filter(Language == language)
    data = fullData %>% filter(Language == language)
    plot = ggplot(data, aes(x=Memory, y=UnigramCE-MedianEmpirical, fill=Type, color=Type)) 
    plot = plot + geom_line(size=2)
    plot = plot + theme_classic() + theme(legend.position="none") 
    plot = plot + geom_line(aes(x=Memory, y=UnigramCE-MedianLower), linetype="dashed") + geom_line(aes(x=Memory, y=UnigramCE-MedianUpper), linetype="dashed")

    quantiles = data.frame(Memory = c(), MI = c(), Density = c())
    if(nrow(dataR) > 0) {
        for(i in (1:max(dataR$Position))) {
           dataRP = dataR  %>% filter(Position == i )
           quants = quantile(dataRP$Surprisal, (1:9)/10)
           dens = data.frame(MI=quants, Quantile=(1:9)/10)
           quantiles = rbind(quantiles, dens %>% mutate(Memory = dataRP$Memory[[1]]))
        }
        quantiles$Type = "RANDOM_BY_TYPE"
        plot = plot + geom_line(data = quantiles, aes(x=Memory, y=UnigramCE-MI, group=Quantile), linetype = "dotted", size=0.5) 
    }
    # plot = plot + geom_density2d(data=dataR, aes(x=Memory, y=UnigramCE-Surprisal))
    ggsave(plot, file=paste("figures/",language,"-listener-surprisal-memory-MEDIANS_QUANTILES_onlyWordForms_boundedVocab.pdf", sep=""))
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
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Russian")
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Erzya-Adap")
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
plot = memListenerSurpPlot_onlyWordForms_boundedVocab("Korean")



