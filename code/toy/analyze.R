library(tidyr)
library(dplyr)
library(ggplot2)

data = read.csv("results.tsv", sep="\t")

data = data %>% mutate(Order = ifelse(Type=="good", "A", "B"))


plot = ggplot(data=data, aes(x=Distance, y=MI, color=Order)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))

ggsave("figures/toy-mis.pdf")




plot = ggplot(data=data, aes(x=Distance, y=Distance * MI, color=Order)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))

ggsave("figures/toy-t-mis.pdf")



data$MI[is.na(data$MI)] = 0
data = data %>% group_by(Order) %>% mutate(CumMI = cumsum(MI), Memory = cumsum(Distance*MI)) %>% mutate(Surprisal = 2.07-CumMI)
plot = ggplot(data=data %>% filter(Surprisal < 2.0), aes(x=Surprisal, y=Memory, color=Order)) + geom_line(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))


ggsave("figures/toy-surp-mem.pdf")


plot = ggplot(data=data %>% filter(Surprisal < 2.0), aes(x=Memory, y=Surprisal, color=Order)) + geom_line(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))


ggsave("figures/toy-mem-surp.pdf")











