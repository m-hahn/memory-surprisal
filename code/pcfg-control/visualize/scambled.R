i1=c(0, 0.2721713981628433, 0.030851562499997875, -0.0030195312499987637, -0.00011523437499860734, 0.00030468749999812417, -0.0016074218750006253, 0.0002031249999987494)
i2=c(0, 0.152348644256592, 0.13503564453124994, 0.012028808593750107, -0.0027910156249992824, 0.0030908203124990763, -0.001051757812499332, -0.0021035156250022177)


unigramEnt = 4.5

surprisals1 = unigramEnt - cumsum(i1)
surprisals2 = unigramEnt - cumsum(i2)

memories1 = cumsum((0:7)*i1)
memories2 = cumsum((0:7)*i2)


data1 = data.frame(surprisal=surprisals1, memory=memories1)
data2 = data.frame(surprisal=surprisals2, memory=memories2)
data1$Type = "MLE"
data2$Type = "Scrambled"
data = rbind(data1, data2)

library(ggplot2)

plot = ggplot(data, aes(x=memory, y=surprisal, group=Type, color=Type)) + geom_line()




