data = read.csv("results.tsv", sep="\t")

library(tidyr)
library(ggplot2)
library(dplyr)

data = data %>% filter(grepl("_2.py", Script))


unigramSurprisal = data %>% filter(Distance == 0) %>% rename(UnigramSurprisal = Surprisal) %>% select(Language, Model, UnigramSurprisal, Estimator)
unigramSurprisal$Distance=NULL
data = merge(data, unigramSurprisal)



plot = ggplot(data %>% filter(Estimator == "Plugin", Model != "GROUND", Distance==0), aes(x=Model, y=It, color=Model, fill=Model)) + geom_col() + facet_wrap(~ Language + Estimator)

plot = ggplot(data %>% filter(Estimator == "Plugin", Model != "GROUND", Distance==1), aes(x=Model, y=It, color=Model, fill=Model)) + geom_col() + facet_wrap(~ Language + Estimator)

plot = ggplot(data %>% filter(Estimator == "Plugin", Model != "GROUND") %>% group_by(Model, Language, SumTIt, UnigramSurprisal) %>% summarise(SumIt = mean(SumIt)), aes(x=SumTIt, y=UnigramSurprisal-SumIt, color=Model, fill=Model, group=Model)) + geom_line() + facet_wrap(~ Language)



#####################################################################################################################################

data = read.csv("resultsByWord.tsv", sep="\t")

library(tidyr)
library(ggplot2)
library(dplyr)

data = data %>% filter(grepl("_2.py", Script))


unigramSurprisal = data %>% filter(Distance == 0) %>% rename(UnigramSurprisal = Surprisal) %>% select(Language, Part, Model, UnigramSurprisal)
unigramSurprisal$Distance=NULL
data = merge(data, unigramSurprisal)


plot = ggplot(data %>% filter(Estimator == "Plugin", Model != "GROUND", Distance==0, Part != "case", Part != "EOS"), aes(x=Model, y=It, color=Model, fill=Model)) + geom_col() + facet_wrap(~ Language + Part)

plot = ggplot(data %>% filter(Estimator == "Plugin", Model != "GROUND", Distance==1, Part != "case", Part != "EOS"), aes(x=Model, y=It, color=Model, fill=Model)) + geom_col() + facet_wrap(~ Language + Part)

plot = ggplot(data, aes(x=Distance, y=it, color=Model)) + geom_line() + facet_wrap(~ Language + Estimator + Model)




