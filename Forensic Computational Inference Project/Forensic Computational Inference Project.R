library(ggplot2)
library(hrbrthemes)
library(GGally)
library(viridis)
library(randomForest)
library(mixture)
library(e1071)
library(readr)
theme_set(theme_minimal())


library(skimr)

skim(glass)

my_skim <- skim_with(
  numeric = sfl(Min = min, p25 = ~ quantile(., probs = .25),Mean=mean,Median=median,
                p75 = ~ quantile(., probs = .75), Max=max,Standard_Deviation=sd,iqr = IQR),
  append=FALSE
)
rr = my_skim(glass)
my_skim(glass)%>%
  yank("numeric")

rr <- dplyr::mutate(rr,n_missing=NULL,complete_rate=NULL,Variable=skim_variable)
rr



glass <- read_csv("glass.csv", col_names = c("Id","RI","Na","Mg","Al","Si","K","Ca",
                                             "Ba","Fe","Type"))


glass_data = as.data.frame(glass)

for (c in 1:length(glass_data$Type)) {
  if (glass_data$Type[c] == 1) {
    glass_data$Type[c] = "Building Windows FP"
  }
  if (glass_data$Type[c] == 2) {
    glass_data$Type[c] = "Building Windows"
  }
  if (glass_data$Type[c] == 3) {
    glass_data$Type[c] = "Vehicle Windows FP"
  }
  if (glass_data$Type[c] == 4) {
    glass_data$Type[c] = "Vehicle Windows"
  }
  if (glass_data$Type[c] == 5) {
    glass_data$Type[c] = "Containers"
  }
  if (glass_data$Type[c] == 6) {
    glass_data$Type[c] = "Tableware"
  }
  if (glass_data$Type[c] == 7) {
    glass_data$Type[c] = "Headlamps"
  }
}
glass_data$Type<-factor(glass_data$Type)


ggparcoord(glass_data,
           columns = 2:10, groupColumn = 11,
           scale = "uniminmax", 
           alphaLines = 0.14,
           boxplot = TRUE,
           title = "Parallel Coord. Plot of Glass Type Predictor Variables") +
  facet_wrap(~ Type) +
  theme(
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 9),
    legend.position="top",
    plot.title = element_text(hjust = 0.5),
    plot.margin = unit(c(0.2, 0.5, 0.2, 0.5), "cm")
  )




library(corrplot)
corrplot(cor(glass_data[,2:10]), type = "upper",tl.col = "black", tl.srt = 45)







library(randomForest)
library(mixture)
library(e1071)
library(readr)

glass <- read_csv("glass.csv", col_names = c("Id","RI","Na","Mg","Al","Si","K","Ca",
                                             "Ba","Fe","Type"))

glass_data = as.data.frame(glass)

for (c in 1:length(glass_data$Type)) {
  if (glass_data$Type[c] == 1) {
    glass_data$Type[c] = "Building Windows FP"
  }
  if (glass_data$Type[c] == 2) {
    glass_data$Type[c] = "Building Windows"
  }
  if (glass_data$Type[c] == 3) {
    glass_data$Type[c] = "Vehicle Windows FP"
  }
  if (glass_data$Type[c] == 4) {
    glass_data$Type[c] = "Vehicle Windows"
  }
  if (glass_data$Type[c] == 5) {
    glass_data$Type[c] = "Containers"
  }
  if (glass_data$Type[c] == 6) {
    glass_data$Type[c] = "Tableware"
  }
  if (glass_data$Type[c] == 7) {
    glass_data$Type[c] = "Headlamps"
  }
}



set.seed(1)
glass_data$Type<-factor(glass_data$Type)
train = sample (1: nrow(glass_data), nrow(glass_data)/2)
glass_data.test=glass_data[-train,"Type"]
glass_data[,1:10]<-scale(glass_data[,1:10])

set.seed(1)
rf.glass_data = tune.randomForest(Type~., data = glass_data[train,], mtry = 5:10,ntree=10*1:25,tunecontrol = tune.control(sampling = "cross",cross=25))
summary(rf.glass_data)
plot(rf.glass_data)

set.seed(1)
rf.glass_data_selected<-randomForest(Type~.,data = glass_data[train,],mtry=6,ntree=40,importance=TRUE,type="class")
rf.glass_data

glass_data.pred=predict(rf.glass_data_selected,glass_data[-train,],type="class")
tab = table(glass_data[-train,"Type"],glass_data.pred)

tab
1-classAgreement(tab)$diag
classAgreement(tab)$crand



plot(rf.glass_data)



library(cvms)
library(broom)  
library(tibble)

tt = tibble("Target" = glass_data[-train,"Type"],
            "Prediction" = glass_data.pred)
set.seed(1)

tab_tt <- table(tt)
cfm <- tidy(tab_tt)



plot_confusion_matrix(cfm, 
                      target_col = "Target", 
                      prediction_col = "Prediction",
                      counts_col = "n") +
  theme(text = element_text(size = 7))





glass_data = as.matrix(glass)

set.seed(5)
glass_predictors = scale(glass_data[,-11])
set.seed(5)
glass_clust = gpcm(glass_predictors, G=6, start=0, atol=1e-2)

tab <- table(glass_data[,11],glass_clust$map)