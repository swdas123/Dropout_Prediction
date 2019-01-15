train=read.csv("/home/lenovo/Downloads/RSTUDIO_WORKSPACE/ML_NEW/CLASSIFICATION/DROPOUT-PREDICTION/dropout_train.csv")
test=read.csv("/home/lenovo/Downloads/RSTUDIO_WORKSPACE/ML_NEW/CLASSIFICATION/DROPOUT-PREDICTION/dropout_test.csv")
class(train) #dataframe
dim(train)
dim(test)
#--------------------------
trainF=train[-c(1:3)]
testF=test[-c(1:3)]
summary(trainF)
train1=lapply(trainF,function(x) ifelse(is.na(x) , mean(x,na.rm = TRUE), x))
test1=lapply(testF,function(x) ifelse(is.na(x),mean(x,na.rm=TRUE),x))
# dim(train1)  #NULL SINCE its a LIST
train2=data.frame(train1)
# dim(train1)
train2$dropout=as.factor(train1$dropout) #since c5.0 requires factor outcome
test2=data.frame(test1) #since PREDICT takes dataframes
#------------------------------
library(C50)
model=C5.0(train2[-17],train2$dropout)
pred=predict(model,test2)
library(caret)
confusionMatrix(table(test2$dropout,pred),mode="everything") #accuracy=87.54%,kappa=0.5704
#----------------------------------

library("gbm")
gbmodel=gbm(dropout ~ ., data = train1, distribution = "bernoulli",shrinkage = 0.3,n.trees = 100,
            interaction.depth = 5)
pred1=predict(gbmodel,test1,n.trees = 80) # here PREDICT doesnot take dataframe
pred1
pred1=ifelse(pred1>0.5,1,0)
pred1
class(test1$dropout)
# test$dropout=as.factor(test$dropout)
# test$dropout
# pred1=as.factor(pred1)
confusionMatrix(table(pred1,test1$dropout),mode = "everything") #accuracy=87.21%,kappa=0.5908





