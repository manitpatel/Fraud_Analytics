#importing the data####

library(readr)
Credit_Fraud_Data <- read_csv("C:/Users/manit_patel/Downloads/Spring 18/R SCRIPTS REPOSITORY/KAGGLE PROJECTS/creditcard_fraud_detection.csv/PS_20174392719_1491204439457_log.csv/Credit_Fraud_Data.csv")
View(Credit_Fraud_Data)

#checking for missing values####

colSums(is.na(Credit_Fraud_Data))

table(Credit_Fraud_Data$isFraud)

#Exploratory Data Analysis####

#Univariate analysis: Type of the transaction
table(Credit_Fraud_Data$type)

Fraud_data=Credit_Fraud_Data[Credit_Fraud_Data$isFraud==1,]

table(Fraud_data$type)

#We find that of the five types of transactions, fraud occurs only in two of them 'TRANSFER' where money is sent to a customer / fraudster and 'CASH_OUT' where money is sent to a merchant who pays the customer / fraudster in cash. 
#Remarkably, the number of fraudulent TRANSFERs almost equals the number of fraudulent CASH_OUTs

#Univariate analysis: isFlaggedFraud

table(Credit_Fraud_Data$isFlaggedFraud)

#only 16 observations are flagged and they are accurately flagged

Flagged_data=Credit_Fraud_Data[Credit_Fraud_Data$isFlaggedFraud==1,]

mean(Flagged_data$amount)

mean(Credit_Fraud_Data$amount)

quantile(Credit_Fraud_Data$amount)

hist(Credit_Fraud_Data$amount)


Flagged_Transfer_amount=Credit_Fraud_Data[Credit_Fraud_Data$type=="TRANSFER" & Credit_Fraud_Data$amount > 4861598, ]

#So, we are not able to see a clear pattern of what determines the flag to to be 1

#Can oldBalanceDest and newBalanceDest determine isFlaggedFraud being set? The old is identical to the new balance 
#in the origin and destination accounts, for every TRANSFER where isFlaggedFraud is set.This is presumably because the transaction is halted 

#Interestingly, oldBalanceDest = 0 in every such transaction. \
#However, as shown below, since isFlaggedFraud can remain not set in TRANSFERS where oldBalanceDest and newBalanceDest can both be 0, these conditions do not determine the state of isFlaggedFraud.

#It can be easily seen that transactions with isFlaggedFraud set occur at all values of step, similar to the complementary set of transactions. Thus isFlaggedFraud does not correlate with step either and is therefore seemingly unrelated to any explanatory variable or feature in the data

#Conclusion: Although isFraud is always set when isFlaggedFraud is set, since isFlaggedFraud is set just 16 times in a seemingly meaningless way, we can treat this feature as insignificant and discard it in the dataset without loosing information.


##### Creating a balanced Data of the frauds 

# randomly selecting 500000 rows which are not fraud and combining them with all 8000 frauds. So it is a small and balanced dataset to do analysis
library(dplyr)


Random= sample_n(Credit_Fraud_Data, 100000)

Fraud_data=Credit_Fraud_Data[Credit_Fraud_Data$isFraud==1,]

table(Random$isFraud)

Random =Random[ !(Random$isFraud==1),]


class(Random$isFraud)
class(Fraud_data$isFraud)


#combining the random and the fraud data now to create the final balanced data

Final_balanced_data = bind_rows(Random, Fraud_data)


# NOW WE HAVE THE FINAL BALANCED DATA 

table(Final_balanced_data$isFraud)



#Feature Engineering

Final_balanced_data$ORI_AMT= Final_balanced_data$oldbalanceOrg/Final_balanced_data$amount

Final_balanced_data$Error_bal_origin= Final_balanced_data$newbalanceOrig + Final_balanced_data$amount - Final_balanced_data$oldbalanceOrg
Final_balanced_data$Error_bal_dest= Final_balanced_data$oldbalanceDest + Final_balanced_data$amount -Final_balanced_data$newbalanceDest

#Dependent variable

class(Final_balanced_data$isFraud)

Final_balanced_data$isFraud=as.factor(as.integer(Final_balanced_data$isFraud))


#Splitting the data ####
ratio = sample(1:nrow(Final_balanced_data), size = 0.4*nrow(Final_balanced_data))
Test = Final_balanced_data[ratio,] #Test dataset 40% of total
Training = Final_balanced_data[-ratio,] #Train dataset 60% of total


#Model 1: Logistic Regression

attach(Training)
Log_model=glm(isFraud~  amount + type+ ORI_AMT+ Error_bal_origin +Error_bal_dest , data = Training, family = "binomial")
summary(Log_model)


#Accuracy of log model on test data
predict_Log_test=predict(Log_model, type="response", newdata=Test)
table(Test$isFraud,predict_Log_test>0.5)

#Calculating c-stat on Test data
library(ROCR)
pred_input_test=prediction(predict_Log_test,Test$isFraud)
AUC= performance(pred_input_test,"auc")
print(AUC@y.values)




###### MODEL 2 : DECISION TREES 
library(rpart)
library(rpart.plot)

CART_model=rpart(isFraud~step+ amount + ORI_AMT+ type +oldbalanceOrg+  newbalanceDest + Error_bal_origin +Error_bal_dest , data=Training, method="class")
prp(CART_model)
summary(CART_model)


#Accuracy of CART model on test data

predict_CART_test=predict(CART_model,newdata=Test, type="class")
table(Test$isFraud,predict_CART_test)
(1346+66)/nrow(Test)

#AUC: As the dependent variable is imbalanced, AUC should be evaluated instead of Accuracy   Calculating c-stat on Test data
pred_CART_test=predict(CART_model, newdata=Test)
pred_prob_Test_CART=pred_CART_test[, 2]

library(ROCR)

pred_input_test_CART=prediction(pred_prob_Test_CART,Test$isFraud)
AUC= performance(pred_input_test_CART,"auc")
print(AUC@y.values)



#MODEL 3: Bagging

#load libraries 
library(data.table)
library(BBmisc)
library(mlr)
library(h2o)


setDT(Training)
setDT(Test)


#Being a binary classification problem, you are always advised to check if the data is imbalanced or not.
setDT(Training)[,.N/nrow(Training),isFraud]
setDT(Test)[,.N/nrow(Test),isFraud]

#Before we start model training, we should convert all character variables to factor
fact_col <- colnames(Training)[sapply(Training,is.character)]
for(i in fact_col)
  set(Training,j=i,value = factor(Training[[i]]))

for(i in fact_col)
  set(Test,j=i,value = factor(Test[[i]]))


#Let's start with modeling now. MLR package has its own function to convert data into a task, build learners, and optimize learning algorithms.

#create a task
traintask <- makeClassifTask(data = Training,target = "isFraud") 
testtask <- makeClassifTask(data = Test,target = "isFraud")

#create learner
bag <- makeLearner("classif.rpart",predict.type = "response")
bag.lrn <- makeBaggingWrapper(learner = bag,bw.iters = 100,bw.replace = TRUE)


#I've set up the bagging algorithm which will grow 100 trees on randomized samples of data with replacement. To check the performance, let's set up a validation strategy 

#set 5 fold cross validation
rdesc <- makeResampleDesc("CV",iters=5L)

#For faster computation, we'll use parallel computation backend.
library(parallelMap)
library(parallel)
parallelStartSocket(cpus = detectCores())

r <- resample(learner = bag.lrn , task = traintask, resampling = rdesc, measures = list(tpr,fpr,fnr,fpr,acc) ,show.info = T)

#With 100 trees, bagging has returned an accuracy of 77%


####Model 4: Random Forest ########

library(randomForest)


fit <- randomForest(isFraud ~ step+ amount + type + ORI_AMT+oldbalanceOrg+  newbalanceDest + Error_bal_origin +Error_bal_dest, Training,ntree=500)

summary(fit)

#Predict Output

predicted= predict(fit, newdata=Test, type="class")

predicted= as.numeric(as.factor(predicted))
table(Test$isFraud, predicted > 0.5)



# Checking amount as a predictor 

mean(Fraud_data$amount)
mean(Credit_Fraud_Data$amount)

min(Fraud_data$amount)





## K-means segmentation of Fraud Data 
#only keeping required data in new dataset

cluster_data= Fraud_data[, -c(1,4,7,10,11,12,13)]

DATAMatrix=as.matrix(cluster_data)

dailykosVector=as.vector(cluster_data)

DATAVector=as.vector(cluster_data)
 
distance=dist(DATAVector,method = "euclidean")

clusterIntensity=hclust(distance, method="ward.D")

plot(clusterIntensity)

#decide the number of clusters from the dendograms

ClusterGroups=cutree(clusterIntensity,k=3)

Cluster1=subset(DATAVector,ClusterGroups==1)
Cluster2=subset(DATAVector,ClusterGroups==2)
Cluster3=subset(DATAVector,ClusterGroups==3)


summary(Cluster1)
summary(Cluster2)
summary(Cluster3)

#Clusters are made primarily on the amount.
