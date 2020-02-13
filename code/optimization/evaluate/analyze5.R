library(dplyr)
library(tidyr)
language = "Basque"
for(file in list.files("~/scr/CODE/memory-surprisal/results/manual_output_ground_coarse/")) {
   if(grepl(language, file)) {
      result = file
      break
   }
}
data_ground = read.csv(paste("~/scr/CODE/memory-surprisal/results/manual_output_ground_coarse/", result, sep=""), sep="\t")
data_ground = data_ground %>% rename(CoarseDependency = Dependency)
PATH = "/u/scr/mhahn/deps/locality_optimized_i1"
data_total = data.frame()
files = c()
corrP = c()
for(file in list.files(PATH)) {
   if(grepl(paste("#", language), paste("#", file))) {
     data = read.csv(paste(PATH, "/", file, sep=""), sep="\t")
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
values = data.frame(Model=files, corrP=corrP)
if(nrow(values)>0) {
  values[order(-values$corrP),]
}

resu = read.csv(paste("output/", language, ".tsv", sep=""), sep="\t")

values = merge(resu, values, by=c("Model"), all=TRUE)

data_mean = merge(data_total %>% group_by(CoarseDependency) %>% summarise(DistanceWeight = median(DistanceWeight)), data_ground %>% select(CoarseDependency, Distance_Mean_NoPunct), by=c("CoarseDependency"))
cat(paste(cor(data_mean$DistanceWeight, data_mean$Distance_Mean_NoPunct), "\n"))
data_mean = data_mean[order(data_mean$DistanceWeight),]

data_bin = data_mean 

PATH = "/u/scr/mhahn/deps/locality_optimized_dlm/manual_output_funchead_coarse_depl/"
data_total = data.frame()
files = c()
corrP = c()
for(file in list.files(PATH)) {
   if(grepl(paste("#", language), paste("#", file))) {
     data = read.csv(paste(PATH, "/", file, sep=""), sep="\t")
     data_b = merge(data, data_ground, by=c("CoarseDependency"))
     pValue = cor.test(data_b$DistanceWeight, data_b$Distance_Mean_NoPunct)$p.value
     cat(file, " ", pValue, "\n")
     data$Model = file
     data_total = rbind(data_total, data)
     files = c(files, file)
     corrP = c(corrP, pValue)
   }
}
values = data.frame(Model=files, corrP=corrP)
if(nrow(values)>0) {
  values[order(-values$corrP),]

  resu = read.csv(paste("output/", language, ".tsv", sep=""), sep="\t")
  
  values = merge(resu, values, by=c("Model"), all=TRUE)
  
  data_mean = merge(data_total %>% group_by(CoarseDependency) %>% summarise(DistanceWeight = median(DistanceWeight)), data_ground %>% select(CoarseDependency, Distance_Mean_NoPunct), by=c("CoarseDependency"))
  cat(paste(cor(data_mean$DistanceWeight, data_mean$Distance_Mean_NoPunct), "\n"))
  data_mean = data_mean[order(data_mean$DistanceWeight),]

  data_d = data_mean

}


PATH = "/u/scr/mhahn/deps/locality_optimized_neural/manual_output_funchead_langmod_coarse_best_ud/"
data_total = data.frame()
files = c()
corrP = c()
for(file in list.files(PATH)) {
   if(grepl(paste("#", language), paste("#", file))) {
     data = read.csv(paste(PATH, "/", file, sep=""), sep="\t")
     data_b = merge(data, data_ground, by=c("CoarseDependency"))
     pValue = cor.test(data_b$DistanceWeight, data_b$Distance_Mean_NoPunct)$p.value
     cat(file, " ", pValue, "\n")
     data$Model = file
     data_total = rbind(data_total, data)
     files = c(files, file)
     corrP = c(corrP, pValue)
   }
}
data_n = NULL
values = data.frame(Model=files, corrP=corrP)
if(nrow(values)>0) {
  values[order(-values$corrP),]
  resu = read.csv(paste("output/", language, ".tsv", sep=""), sep="\t")

  values = merge(resu, values, by=c("Model"), all=TRUE)
  
  data_mean = merge(data_total %>% group_by(CoarseDependency) %>% summarise(DistanceWeight = median(DistanceWeight)), data_ground %>% select(CoarseDependency, Distance_Mean_NoPunct), by=c("CoarseDependency"))
  cat(paste(cor(data_mean$DistanceWeight, data_mean$Distance_Mean_NoPunct), "\n"))
  data_mean = data_mean[order(data_mean$DistanceWeight),]
  
  
  data_n = data_mean
}

data = merge(data_bin %>% select(CoarseDependency, DistanceWeight, Distance_Mean_NoPunct) %>% rename(DistanceWeight_b = DistanceWeight), data_d %>% select(CoarseDependency, DistanceWeight) %>% rename(DistanceWeight_d = DistanceWeight), by=c("CoarseDependency"))
if(!is.null(data_n)) {
  data = merge(data, data_n %>% select(CoarseDependency, DistanceWeight) %>% rename(DistanceWeight_n = DistanceWeight), by=c("CoarseDependency"))
  summary(lm(Distance_Mean_NoPunct ~ DistanceWeight_b + DistanceWeight_d + DistanceWeight_n, data=data))
}
summary(lm(Distance_Mean_NoPunct ~ DistanceWeight_b + DistanceWeight_d, data=data))

cor.test(data$DistanceWeight_b, data$Distance_Mean_NoPunct, method="spearman")


