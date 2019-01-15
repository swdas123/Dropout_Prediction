
train=read.csv("/home/lenovo/Downloads/RSTUDIO_WORKSPACE/ML_NEW/CLASSIFICATION/DROPOUT-PREDICTION/dropout_train.csv")
test=read.csv("/home/lenovo/Downloads/RSTUDIO_WORKSPACE/ML_NEW/CLASSIFICATION/DROPOUT-PREDICTION/dropout_test.csv")
str(train)
class(train)
dim(train)
summary(train)

#---------for DECISION TREE
trainF=train[-c(1:3)]
testF=test[-c(1:3)]
train1=data.frame(lapply(trainF, function(x) ifelse (is.na(x),mean(x,na.rm=TRUE),x)))
test1=data.frame(lapply(testF,function(x) ifelse(is.na(x),mean(x,na.rm = TRUE),x)))
dim(train1)

train1$dropout=as.factor(train1$dropout)
test1$dropout=as.factor(test1$dropout)
library(C50)
model=C5.0(train1[-17],train1$dropout)
pred=predict(model,test1)
class(pred)
library(caret)
confusionMatrix(pred,test1$dropout,mode="everything",positive = '1')
#Accuracy : 0.8754, Kappa : 0.5704 

#--------------------FOR GRADIENT BOOST

train2F=train[-c(1:3)]
test2F=test[-c(1:3,20)]
train2=data.frame(lapply(train2F,function(x) ifelse (is.na(x),mean(x,na.rm = TRUE),x)))
test2=data.frame(lapply(test2F,function(x) ifelse (is.na(x),mean(x,na.rm = TRUE),x)))

library(gbm)
gbmodel=gbm(dropout ~ . , data=train2, distribution = "bernoulli", shrinkage = 0.3,n.cores = NULL,interaction.depth = 5, n.trees = 100)
pred2=predict(gbmodel,test2,n.trees = 80)
pred2=ifelse(pred2>0.5,1,0)
class(pred2)
pred2=as.factor(pred2)
confusionMatrix(pred2,test1$dropout,positive = '1')
#Accuracy : 0.8732, Kappa : 0.5956 

#------------
library("ROCR")

class(pred)  #factor 1,0
class(test1$dropout)#factor 1,0

pred=as.data.frame(pred)
dim(pred)
class(pred)
test1$dropout=as.data.frame(test1$dropout)
dim(test1$dropout)
test1$dropout
class(test1$dropout)


cc<- function(x) {
  x <- ifelse(x >0, 1, 0)
}
# MARGIN = 2, apply on column


pred1 <- apply(pred, MARGIN = 1,
                       cc)
test1$dropout1 <- apply(test1$dropout, MARGIN = 1,
                               cc)

class(pred1)
class(test1$dropout1)
pred3 <- prediction(predictions = pred1,labels = test1$dropout1)
perf <- performance(pred3, measure = "tpr", x.measure = "fpr")
plot(perf)
# Plot precision/recall curve
perf <- performance(pred3, measure = "prec", x.measure = "rec")
plot(perf)
# Plot accuracy as function of threshold
perf <- performance(pred3, measure = "acc")
plot(perf)
