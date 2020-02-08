library(dplyr)
library(tidyr)
data_dlm = read.csv("/u/scr/mhahn/deps/locality_optimized_dlm/manual_output_funchead_coarse_depl/English_optimizeDependencyLength.py_model_3137767.tsv", sep="\t")
data_i1 = read.csv("/u/scr/mhahn/deps/locality_optimized_i1/English_optimizeGrammarForI1_6_Long.py_model_876472065.tsv", sep="\t")
data_neural = read.csv("/u/scr/mhahn/deps/locality_optimized_neural/manual_output_funchead_langmod_coarse_best_ud/English_optimizePredictability.py_model_2194458393.tsv", sep="\t")
data_ground = read.csv("~/scr/CODE/memory-surprisal/results/manual_output_ground_coarse/English_inferWeightsCrossVariationalAllCorpora_NoPunct_NEWPYTORCH_Coarse.py_model_3723683.tsv", sep="\t")
data_ground = data_ground %>% rename(CoarseDependency = Dependency)

data_b = merge(data_i1, data_ground, by=c("CoarseDependency"))
data_n = merge(data_neural, data_ground, by=c("CoarseDependency"))
data_d = merge(data_dlm, data_ground, by=c("CoarseDependency"))

cor.test(data_b$DistanceWeight, data_b$Distance_Mean_NoPunct)$p.value
cor.test(data_n$DistanceWeight, data_n$Distance_Mean_NoPunct)$p.value
cor.test(data_d$DistanceWeight, data_d$Distance_Mean_NoPunct)$p.value
cor.test(data_b$DH_Weight, data_b$DH_Mean_NoPunct)$p.value
cor.test(data_n$DH_Weight, data_n$DH_Mean_NoPunct)$p.value
cor.test(data_d$DH_Weight, data_d$DH_Mean_NoPunct)$p.value


