library(tidyr)
library(dplyr)
library(ggplot2)

data = read.csv("results.tsv", sep="\t")

data = data %>% mutate(Order = ifelse(Type=="good", "A", "B"))


data = data %>% mutate(Distance_ = ifelse(Order == "A" & Distance == 1.0, 1.3, Distance))

plot = ggplot(data=data, aes(x=Distance_, y=MI, color=Order)) + 
	geom_step(data=data %>% filter(Order == "A"), size=2, linetype=2) + 
	geom_step(data=data %>% filter(Order == "B"), size=2, linetype=2) + 
	geom_step(data=data %>% filter(Distance_==2 | Distance_==3), size=2, linetype=1) + 
	geom_step(data=data %>% filter(Distance_==5 | Distance_==6), size=2, linetype=1) + 
	theme_bw() + 
	theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) +
	xlab("t") + ylab("Conditional Mutual Information (It)")

ggsave("figures/toy-mis.pdf")




plot = ggplot(data=data, aes(x=Distance_, y=Distance * MI, color=Order)) + 
	geom_step(size=2, linetype=2) + 
	geom_step(data=data %>% filter(Distance_==2 | Distance_==3), size=2, linetype=1) + 
	geom_step(data=data %>% filter(Distance_==5 | Distance_==6), size=2, linetype=1) + 
	theme_bw() + 
	theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) +
	xlab("t") + ylab("t * It")

ggsave("figures/toy-t-mis.pdf")



data$MI[is.na(data$MI)] = 0
data = data %>% group_by(Order) %>% mutate(CumMI = cumsum(MI), Memory = cumsum(Distance*MI)) %>% mutate(Surprisal = 2.07-CumMI)
#plot = ggplot(data=data %>% filter(Surprisal < 2.0), aes(x=Surprisal, y=Memory, color=Order)) + geom_line(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))
#ggsave("figures/toy-surp-mem.pdf")

data = data %>% mutate(Surprisal_ = ifelse(Order == "A" & Memory < 2, 0.285, Surprisal))
data = data %>% mutate(Memory_ = ifelse(Order == "A" & Memory < 2, 1.79, Memory))


data_ = data %>% select(Surprisal_, Memory_, Order)

data_ = rbind(as.data.frame(data_), data.frame(Memory_=c(2.07, 2.433962), Surprisal_=c(0.146, 0.07441896), Order=c("A", "A")))

plot = ggplot(data=data_ %>% filter(Surprisal_ < 2.0), aes(x=Memory_, y=Surprisal_, color=Order)) + 
	geom_line(size=2, linetype=2) + 
	geom_line(data=data_%>%filter(Memory_>2.0), size=2, linetype=1) + 
	theme_bw() + 
	theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) +
	xlab("Memory") + ylab("Surprisal")
plot






ggsave("figures/toy-mem-surp.pdf")











