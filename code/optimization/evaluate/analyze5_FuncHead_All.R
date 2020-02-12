library(dplyr)
library(tidyr)

languages = read.csv("/juicier/scr120/scr/mhahn/CODE/grammar-optim/results/languages/languages.tsv", sep="\t")$Language

data_ground = data.frame()
for(language in languages) {
for(file in list.files("~/scr/deps/manual_output_funchead_ground_coarse/")) {
   if(grepl(language, file) & grepl("FuncHead", file)) {
      result = file
      break
   }
}
data_ground_ = read.csv(paste("~/scr/deps/manual_output_funchead_ground_coarse/", result, sep=""), sep="\t")
data_ground_ = data_ground_ %>% rename(CoarseDependency = Dependency) %>% mutate(Language=language)
data_ground = rbind(data_ground, data_ground_)
}
PATH = "/u/scr/mhahn/deps/locality_optimized_i1"
data_total = data.frame()
files = c()
corrP = c()
for(language in languages) {
for(file in list.files(PATH)) {
   if(grepl(paste("#", language), paste("#", file)) & !grepl("Old", file) & grepl("FuncHead", file) & !grepl("POS", file)& grepl("I1_9", file)) {
     data = read.csv(paste(PATH, "/", file, sep=""), sep="\t") %>% mutate(Language = language)
     data_b = merge(data, data_ground, by=c("CoarseDependency"))
     if(data_b$DistanceWeight[1] != "NaN") {
        pValue = cor.test(data_b$DistanceWeight, data_b$Distance_Mean_NoPunct)$p.value
        cat(file, " ", pValue, "\n")
        data$Model = file
        data_total = rbind(data_total, data)
        files = c(files, file)
        corrP = c(corrP, pValue)
     }
   }
}
}
cat("--30\n")
values = data.frame(Model=files, corrP=corrP)
if(nrow(values)>0) {
  values[order(-values$corrP),]
}

#resu = read.csv(paste("output/", language, ".tsv", sep=""), sep="\t")
#values = merge(resu, values, by=c("Model"), all=TRUE)

data_mean = merge(data_total %>% group_by(Language, CoarseDependency) %>% summarise(DistanceWeight = median(DistanceWeight)), data_ground %>% select(Language, CoarseDependency, Distance_Mean_NoPunct), by=c("Language", "CoarseDependency"), all=TRUE)
cat(paste(cor(data_mean$DistanceWeight, data_mean$Distance_Mean_NoPunct), "\n"))
data_mean = data_mean[order(data_mean$DistanceWeight),]

data_bin = data_mean 

PATH = "/u/scr/mhahn/deps/manual_output_funchead_coarse_depl/"
data_total = data.frame()
files = c()
corrP = c()
for(language in languages) {
for(file in list.files(PATH)) {
   if(grepl(paste("#", language), paste("#", file))) {
     data = read.csv(paste(PATH, "/", file, sep=""), sep="\t") %>% rename(CoarseDependency=Dependency) %>% mutate(Language=language)
     data_b = merge(data, data_ground, by=c("CoarseDependency"))
     pValue = cor.test(data_b$DistanceWeight, data_b$Distance_Mean_NoPunct)$p.value
     cat(file, " ", pValue, "\n")
     data$Model = file
     data_total = rbind(data_total, data)
     files = c(files, file)
     corrP = c(corrP, pValue)
   }
}
}
values = data.frame(Model=files, corrP=corrP)
  values[order(-values$corrP),]

#  resu = read.csv(paste("output/", language, ".tsv", sep=""), sep="\t")
  
#  values = merge(resu, values, by=c("Model"), all=TRUE)
  
  data_mean = merge(data_total %>% group_by(Language, CoarseDependency) %>% summarise(DistanceWeight = median(DistanceWeight)), data_ground %>% select(Language, CoarseDependency, Distance_Mean_NoPunct), by=c("Language", "CoarseDependency"), all=TRUE)
  cat(paste(cor(data_mean$DistanceWeight, data_mean$Distance_Mean_NoPunct), "\n"))
  data_mean = data_mean[order(data_mean$DistanceWeight),]

  data_d = data_mean



PATH = "/u/scr/mhahn/deps/manual_output_funchead_langmod_coarse_best/"
data_total = data.frame()
files = c()
corrP = c()
for(language in languages) {
for(file in list.files(PATH)) {
   if(grepl(paste("#", language), paste("#", file))) {
     data = read.csv(paste(PATH, "/", file, sep=""), sep="\t") %>% mutate(Language = language)
     data_b = merge(data, data_ground, by=c("CoarseDependency"))
     pValue = cor.test(data_b$DistanceWeight, data_b$Distance_Mean_NoPunct)$p.value
     cat(file, " ", pValue, "\n")
     data$Model = file
     data_total = rbind(data_total, data)
     files = c(files, file)
     corrP = c(corrP, pValue)
   }
}
}
data_n = NULL
values = data.frame(Model=files, corrP=corrP)
  values[order(-values$corrP),]
#  resu = read.csv(paste("output/", language, ".tsv", sep=""), sep="\t")

#  values = merge(resu, values, by=c("Model"), all=TRUE)
  
  data_mean = merge(data_total %>% group_by(Language, CoarseDependency) %>% summarise(DistanceWeight = median(DistanceWeight)), data_ground %>% select(Language, CoarseDependency, Distance_Mean_NoPunct), by=c("Language", "CoarseDependency"))
  cat(paste(cor(data_mean$DistanceWeight, data_mean$Distance_Mean_NoPunct), "\n"))
  data_mean = data_mean[order(data_mean$DistanceWeight),]
  
  
  data_n = data_mean

data = merge(data_bin %>% select(Language, CoarseDependency, DistanceWeight, Distance_Mean_NoPunct) %>% rename(DistanceWeight_b = DistanceWeight), data_d %>% select(Language, CoarseDependency, DistanceWeight) %>% rename(DistanceWeight_d = DistanceWeight), by=c("Language", "CoarseDependency"), all=TRUE)
  data = merge(data, data_n %>% select(Language, CoarseDependency, DistanceWeight) %>% rename(DistanceWeight_n = DistanceWeight), by=c("Language", "CoarseDependency"), all=TRUE)
  summary(lm(Distance_Mean_NoPunct ~ DistanceWeight_b + DistanceWeight_d + DistanceWeight_n, data=data))
summary(lm(Distance_Mean_NoPunct ~ DistanceWeight_b + DistanceWeight_d, data=data))

cor.test(data$DistanceWeight_b, data$Distance_Mean_NoPunct, method="spearman")

data_avgs = data %>% group_by(CoarseDependency) %>% summarise(DistanceWeight_b = mean(DistanceWeight_b, na.rm=TRUE), DistanceWeight_n = mean(DistanceWeight_n, na.rm=TRUE), DistanceWeight_d = mean(DistanceWeight_d, na.rm=TRUE), Distance_Mean_NoPunct = mean(Distance_Mean_NoPunct, na.rm=TRUE))


#> cor(data_avgs$DistanceWeight_b, data_avgs$Distance_Mean_NoPunct)
#[1] 0.6699087
#> cor(data_avgs$DistanceWeight_n, data_avgs$Distance_Mean_NoPunct)                                                                                                                                         #[1] 0.6178542
#> cor(data_avgs$DistanceWeight_d, data_avgs$Distance_Mean_NoPunct)                                                                                                                                         
#[1] 0.6800993
#> cor(data_avgs$DistanceWeight_d, data_avgs$Distance_Mean_NoPunct, method="spearman")
#[1] 0.7358349
#> cor(data_avgs$DistanceWeight_b, data_avgs$Distance_Mean_NoPunct, method="spearman")
#[1] 0.7196998
#> cor(data_avgs$DistanceWeight_n, data_avgs$Distance_Mean_NoPunct, method="spearman")
#[1] 0.6934334
#

print(data_avgs[order(data_avgs$Distance_Mean_NoPunct),], n=50)



