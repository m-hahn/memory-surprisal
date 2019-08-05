library(tidyr)
library(dplyr)
library(ggplot2)

data = read.csv("results.tsv", sep="\t")

data = data %>% mutate(Order = ifelse(Type=="good", "A", "B"))


plot = ggplot(data=data, aes(x=Distance, y=MI, color=Order)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))

ggsave("toy-mis.pdf")




plot = ggplot(data=data, aes(x=Distance, y=Distance * MI, color=Order)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))

ggsave("toy-t-mis.pdf")



data$MI[is.na(data$MI)] = 0
data = data %>% group_by(Order) %>% mutate(CumMI = cumsum(MI), Memory = cumsum(Distance*MI)) %>% mutate(Surprisal = 2.07-CumMI)
plot = ggplot(data=data %>% filter(Surprisal < 2.0), aes(x=Surprisal, y=Memory, color=Order)) + geom_line(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))


ggsave("toy-surp-mem.pdf")


plot = ggplot(data=data %>% filter(Surprisal < 2.0), aes(x=Memory, y=Surprisal, color=Order)) + geom_line(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))


ggsave("toy-mem-surp.pdf")







#plot = ggplot(data=data, aes(x=Distance, y=Distance*MI, color=Type)) + geom_line()


x = (1:10)
y = 5 * x^(-2)

data = data.frame(Distance=x, MI=y)

plot = ggplot(data=data, aes(x=Distance, y=MI)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))


plot = ggplot(data=data, aes(x=Distance, y=Distance * MI)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))





x1 = (1:10)
y1 = 5 * x^(-2.5)

data1 = data.frame(Distance=x1, MI=y1)
data1$Process = "A"

x2 = (1:10)
y2 = 3.5 * x^(-1.5)

data2 = data.frame(Distance=x2, MI=y2)
data2$Process = "B"


data = rbind(data1, data2) 

plot = ggplot(data=data, aes(x=Distance, y=MI, color=Process, group=Process)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))



ggsave("decay.pdf")



plot = ggplot(data=data, aes(x=Distance, y=Distance * MI, color=Process, group=Process)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))


ggsave("memory.pdf")







x = c(0.07, 0.07)
y = c(2.2, 2.4)
data = data.frame(Surprisal=x, Memory=y)

plot = ggplot(data=data, aes(x=Surprisal, y=Memory)) + geom_point() + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) + xlim(0,1) + ylim(0,3)


x= c(0.0, 1.0)
     y = c(1.0, 0.0)
data2 = data.frame(Surprisal=x, Memory=y)

plot = ggplot(data=data2, aes(x=Surprisal, y=Memory)) + geom_line() + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) + xlim(0,3) + ylim(0,3) 







x = (1:10)
y = 5 * x^(-2)

data = data.frame(Distance=x, MI=y)

df2 <- rbind(
  data,
  transform(data[order(data$Distance),],
    Distance=Distance - 1e-9,  # required to avoid crazy steps
    MI=ave(MI, FUN=function(z) c(z[[1]], head(z, -1L)))
) )

T = 4
plot = ggplot(data=df2 %>% filter(Distance >= T), aes(x=Distance, y=MI)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) + geom_area() + geom_step(size=2, data=df2 %>% filter(Distance < T+1))
ggsave("add-surp.pdf")


data2 = data.frame(Distance=data$Distance, Memory=data$Distance*data$MI)
df3 <- rbind(
  data2,
  transform(data2[order(data2$Distance),],
    Distance=Distance - 1e-9,  # required to avoid crazy steps
    Memory=ave(Memory, FUN=function(z) c(z[[1]], head(z, -1L)))
) )

plot = ggplot(data=df3 %>% filter(Distance < T), aes(x=Distance, y=Memory)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) + geom_area() + geom_step(size=2, data=df3 %>% filter(Distance >= T))

ggsave("lower-mem.pdf")






for(T in (1:10)) {
   plot = ggplot(data=df2 %>% filter(Distance >= T), aes(x=Distance, y=MI)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) + geom_area() + geom_step(size=2, data=df2 %>% filter(Distance < T+1))
   ggsave(paste("add-surp-",T,".pdf", sep=""))
   
   plot = ggplot(data=df3 %>% filter(Distance < T), aes(x=Distance, y=Memory)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18)) + geom_area() + geom_step(size=2, data=df3 %>% filter(Distance >= T))
   
   ggsave(paste("lower-mem-",T,".pdf", sep=""))
}









x1 = (1:10)
y1 = 5 * x^(-2.5)

data1 = data.frame(Distance=x1, MI=y1)
data1$Process = "A"

data = rbind(data1) 

plot = ggplot(data=data, aes(x=Distance, y=MI)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))



ggsave("decay-A.pdf")



plot = ggplot(data=data, aes(x=Distance, y=Distance * MI)) + geom_step(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))


ggsave("memory-A.pdf")












# Listener's Memory-Surprisal Curve

x1 = (1:10)
y1 = 5.3 * x^(-2.5)

data1 = data.frame(Distance=x1, MI=y1)
data1$Process = "A"

data = rbind(data1) 

library(tidyr)

data = data %>% group_by(Process) %>% mutate(CumMI = cumsum(MI))
data = data %>% group_by(Process) %>% mutate(Memory = cumsum(MI*Distance))

data = data %>% mutate(Surprisal = 10-CumMI)

plot = ggplot(data=data, aes(x=Memory, y=Surprisal)) + geom_line(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))



ggsave("listener-tradeoff-A.pdf")









# Listener's Memory-Surprisal Curve

x1 = (1:10)
y1 = 5.3 * x^(-2.5)

data1 = data.frame(Distance=x1, MI=y1)
data1$Process = "A"

x2 = (1:10)
y2 = 3.5 * x^(-1.5)

data2 = data.frame(Distance=x2, MI=y2)
data2$Process = "B"


data = rbind(data1, data2) 

library(tidyr)

data = data %>% group_by(Process) %>% mutate(CumMI = cumsum(MI))
data = data %>% group_by(Process) %>% mutate(Memory = cumsum(MI*Distance))

data = data %>% mutate(Surprisal = 10-CumMI)

plot = ggplot(data=data, aes(x=Memory, y=Surprisal, color=Process, group=Process)) + geom_line(size=2) + theme_bw() + theme(axis.text=element_text(size=18), axis.title=element_text(size=18), legend.title=element_text(size=18), legend.text=element_text(size=18))



ggsave("listener-tradeoff.pdf")











