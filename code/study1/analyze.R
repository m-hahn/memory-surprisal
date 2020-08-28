library(tidyr)
library(dplyr)
library(ggplot2)

data1 = read.csv("results-simulation-6-short.py.tsv", sep="\t") %>% mutate(Order = "A (Short)")
data2 = read.csv("results-simulation-6-long.py.tsv", sep="\t") %>% mutate(Order = "B (Long)")
data=rbind(data1, data2)

#library(scales)
#show_col(hue_pal()(2))

data = data %>% mutate(Distance_ = ifelse(Order == "A (Short)" & Distance == 1.0, 1.3, Distance)) %>% filter(Distance<7)

#plot = ggplot(data=data, aes(x=Distance, y=MI, color=Order)) + 
#	geom_step(data=data, size=2) + 
#	theme_bw() + 
#	theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) +
#	xlab("t") + ylab("Conditional Mutual Information (It)")
#

data = data %>% mutate(colors=ifelse(Order != "A (Short)", "#F8766D", "#00BFC4"))

plot = ggplot(data=data, aes(x=Distance_, y=1.44*MI, color=Order)) + 
	geom_step(data=data %>% filter(Order == "A (Short)"), size=2, linetype=2) + 
	geom_step(data=data %>% filter(Order == "B (Long)"), size=2, linetype=2) + 
	geom_step(data=data %>% filter(Distance_==2 | Distance_==3), size=2, linetype=1) + 
	geom_step(data=data %>% filter(Distance_==5 | Distance_==6), size=2, linetype=1) + 
	theme_bw() + 
	theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) +
	xlab("t") + ylab("Conditional Mutual Information (It)") + scale_colour_manual(values = c("#00BFC4", "#F8766D"))

ggsave("figures/toy-mis.pdf")




plot = ggplot(data=data, aes(x=Distance_, y=Distance * 1.44 * MI, color=Order)) + 
	geom_step(size=2, linetype=2) + 
	geom_step(data=data %>% filter(Distance_==2 | Distance_==3), size=2, linetype=1) + 
	geom_step(data=data %>% filter(Distance_==5 | Distance_==6), size=2, linetype=1) + 
	theme_bw() + 
	theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) +
	xlab("t") + ylab("t * It") + scale_colour_manual(values = c("#00BFC4", "#F8766D"))

ggsave("figures/toy-t-mis.pdf")



data$MI[is.na(data$MI)] = 0
data = data %>% group_by(Order) %>% mutate(CumMI = cumsum(MI), Memory = cumsum(Distance*MI)) %>% mutate(Surprisal = 1.889-CumMI)
#plot = ggplot(data=data %>% filter(Surprisal < 2.0), aes(x=Surprisal, y=Memory, color=Order)) + geom_line(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))
#ggsave("figures/toy-surp-mem.pdf")

data = data %>% mutate(Surprisal_ = Surprisal) #= ifelse(Order == "A (Short)" & Memory < 2, 0.285, Surprisal))
data = data %>% mutate(Memory_ = Memory) #ifelse(Order == "A (Short)" & Memory < 2, 1.79, Memory))


data_ = data %>% select(Surprisal_, Memory_, Order)

data_ = rbind(as.data.frame(data_), data.frame(Memory_=c(2.12, 2.35), Surprisal_=c(0.077, 0.077), Order=c("A (Short)", "A (Short)")))


plot = ggplot(data=data_ %>% filter(Surprisal_ < 1.0), aes(x=1.44*Memory_, y=1.44*Surprisal_, color=Order)) + 
	geom_line(size=2, linetype=1) + 
	geom_line(data=data_%>%filter(Memory_>1.7), size=2, linetype=1) + 
	theme_bw() + 
	theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) +
	xlab("Memory") + ylab("Surprisal") + scale_colour_manual(values = c("#00BFC4", "#F8766D"))
plot






ggsave("figures/toy-mem-surp.pdf")











