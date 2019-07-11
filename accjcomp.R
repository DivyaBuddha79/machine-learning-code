prosPath = system.file("extdata", "prostate.csv", package = "h2o")
prostate_df <- read.csv(prosPath)
prostate_df <- prostate_df[,-1]
summary(prostate_df)
#splitting into training and testing data
set.seed(1234)
random_splits <- runif(nrow(prostate_df))
train_df <- prostate_df[random_splits < .5,]
dim(train_df)
validate_df <- prostate_df[random_splits >=.5,]
dim(validate_df)
#applying random forest model
install.packages('randomForest')
library(randomForest)

outcome_name <- 'CAPSULE'
feature_names <- setdiff(names(prostate_df), outcome_name)
print(feature_names)
set.seed(1234)
rf_model <- randomForest(x=train_df[,feature_names],
                         y=as.factor(train_df[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")
#calculating auc score for the model
install.packages('pROC')
library(pROC)
auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions[,2])

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')
#using autoencoder for diving into hard and easy sets based on reconstruction error 
library(h2o)
localH2O = h2o.init()
prostate.hex<-as.h2o(train_df, destination_frame="train.hex")
prostate.dl = h2o.deeplearning(x = feature_names, training_frame = prostate.hex,
                               autoencoder = TRUE,
                               reproducible = T,
                               seed = 1234,
                               hidden = c(6,5,6), epochs = 50)
prostate.anon = h2o.anomaly(prostate.dl, prostate.hex, per_feature=FALSE)
head(prostate.anon)
err <- as.data.frame(prostate.anon)
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')
# applying random forest over data split according to reconstruction error < 0.1
train_df_auto <- train_df[err$Reconstruction.MSE < 0.1,]

set.seed(1234)
rf_model <- randomForest(x=train_df_auto[,feature_names],
                         y=as.factor(train_df_auto[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions_known <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions_known[,2])

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')
#applying random forest over data split according to reconstruction error > 0.1
train_df_auto <- train_df[err$Reconstruction.MSE >= 0.1,]

set.seed(1234)
rf_model <- randomForest(x=train_df_auto[,feature_names],
                         y=as.factor(train_df_auto[,outcome_name]),
                         importance=TRUE, ntree=20, mtry = 3)

validate_predictions_unknown <- predict(rf_model, newdata=validate_df[,feature_names], type="prob")

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=validate_predictions_unknown[,2])

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')
#taking equal sets of data from both the splits
valid_all <- (validate_predictions_known[,2] + validate_predictions_unknown[,2]) / 2

auc_rf = roc(response=as.numeric(as.factor(validate_df[,outcome_name]))-1,
             predictor=valid_all)

plot(auc_rf, print.thres = "best", main=paste('AUC:',round(auc_rf$auc[[1]],3)))
abline(h=1,col='blue')
abline(h=0,col='green')
# logistic regression model
set.seed(1234)
lr_model <- glm(CAPSULE ~.,family=binomial(link='logit'),data=train_df)
summary(lr_model)
probabilities <- predict(lr_model, newdata = validate_df, type = "response")
head(probabilities)
y_pred_num <- ifelse(probabilities > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels=c(0, 1))
y_act <-as.factor(validate_df[,outcome_name])
mean(y_pred == y_act)
library(tidyverse)
library(caret)
caret::confusionMatrix(y_pred,y_act,positive="1",mode="everything")
#using linear svm
install.packages('e1071')
library(e1071)
lsvm = svm(formula = CAPSULE ~.,data=train_df,type='C-classification',kernel='linear')
lsvmpredict= predict(lsvm, newdata = validate_df)
mean(lsvmpredict == y_act)
caret::confusionMatrix(lsvmpredict,y_act,positive="1",mode="everything")
# using kernel svm
set.seed(1234)
rsvm = svm(formula = CAPSULE ~.,data=train_df,kernel='radial',cost=1,gamma=1)
rsvmpredict= predict(lsvm, newdata = validate_df)
mean(rsvmpredict == y_act)
#using knn
install.packages('class')
library(class)
set.seed(1234)
ctrl <- trainControl(method="repeatedcv",repeats = 3)
knnFit <- train(CAPSULE ~ ., data = train_df, method = "knn", trControl = ctrl, preProcess = c("center","scale"),tuneLength = 20)
knnFit
plot(knnFit)
knnpred=knn(train = train_df, test = validate_df,cl = train_df$CAPSULE, k=35)
mean(knnpred == y_act)


