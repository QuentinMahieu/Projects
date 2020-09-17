library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)
options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)

#create train and test sets
set.seed(42, sample.kind = "Rounding")
index <- createDataPartition(titanic_clean$Survived, times = 1, p = 0.2)

train_set <- titanic_clean[-index$Resample1, ]
test_set <- titanic_clean[index$Resample1, ]

#proportion of survivors in the train data set
mean(train_set$Survived == 1)
#--------------------------------------------------#
#1st prediction by guessing (chance)
set.seed(3, sample.kind = "Rounding")

sample <- sample(c(0,1), 179, replace = T)
mean(sample== test_set$Survived) #accuracy by guessing

#2 predicting survival by sex
#proportion of female who survived
fem_surv <- train_set %>% filter(Sex == "female") %>%
  summarise(avg = mean(Survived == 1)) %>% .$avg
fem_surv
male_surv <- train_set %>% filter(Sex == "male") %>%
  summarise(avg = mean(Survived == 1)) %>% .$avg
male_surv
#predict survival by sex
sex_model <- ifelse(test_set$Sex == "female", 1, 0)    # predict Survived=1 if female, 0 if male
mean(sex_model == test_set$Survived)    # calculate accuracy

#3 predicting survival by class
train_set %>% ggplot(aes(Pclass, fill = Survived)) +
  geom_bar(position = "dodge", alpha = 0.75)

#4 predicting survival by class
class_model <- ifelse(test_set$Pclass == 1, 1, 0)
mean(class_model == test_set$Survived)

train_set %>% group_by(Sex, Pclass) %>% ggplot(aes())

#4 predicting survival by class & sex
class_sex_model <- ifelse(test_set$Pclass %in% c(1,2)
                          & test_set$Sex == "female", 1, 0)
mean(class_sex_model == test_set$Survived)

#use of the confusion matrix to know the accuracy, specificity, sensitivity and balance accuracy
confusionMatrix(data = factor(sex_model), reference = test_set$Survived)
confusionMatrix(data = factor(class_model),reference = test_set$Survived)
confusionMatrix(data = factor(class_sex_model), reference = test_set$Survived)

#calcul of the f1 score (harmonic average)
F_meas(data = factor(sex_model), reference = test_set$Survived)
F_meas(data = factor(class_model), reference = test_set$Survived)
F_meas(data = factor(class_sex_model), reference = test_set$Survived)

#----------------------------------------------#
# linear discriminant analysis (LDA)
set.seed(1, sample.kind = "Rounding")
fit_lda <- train(Survived ~ Fare, data = train_set, method = "lda")
y_hat_fare <- predict(fit_lda, newdata = test_set)
mean(y_hat_fare == test_set$Survived)

#QDA  model one variable
set.seed(1, sample.kind = "Rounding")
fit_qda <- train(Survived ~ Fare, data = train_set, method = "qda")
y_hat_fare_qda <- predict(fit_qda, newdata = test_set)
mean(y_hat_fare_qda == test_set$Survived)

#QDA multiple 
set.seed(1, sample.kind = "Rounding")
fit_qda_m <- train(Survived ~ Sex + Pclass + Age,
                   data = train_set, method = "qda")
y_hat_qda_m <- predict(fit_qda_m, newdata = test_set)
mean(y_hat_qda_m == test_set$Survived)

#GLM one & all 
set.seed(1, sample.kind = "Rounding")
fit_glm <- train(Survived ~ Sex, data = train_set, method = "glm")
y_hat_sex_glm <- predict(fit_glm, newdata = test_set)
mean(y_hat_sex_glm == test_set$Survived)

set.seed(1, sample.kind = "Rounding")
fit_glm_all <- train(Survived ~ ., data = train_set, method = "glm")
y_hat_sex_glm_all <- predict(fit_glm_all, newdata = test_set)
mean(y_hat_sex_glm_all == test_set$Survived)

#knn model
set.seed(6, sample.kind = "Rounding")
#best neighbors k
fit_knn <- train(Survived ~ ., data = train_set,
                 method = "knn", tuneGrid = data.frame(k = seq(3, 51, 2)))
# plot to vizualise plot(fit_knn) or
fit_knn$bestTune
#use the best k to find the accuracy on the test_set
fit_knn <- train(Survived ~ ., data = train_set,
                 method = "knn", tuneGrid = (k = 11))
y_hat_knn <- predict(fit_knn, newdata = test_set)
mean(y_hat_knn == test_set$Survived)

#knn with cross-validation not defaul(20)
set.seed(8, sample.kind = "Rounding")
fit_knn <- train(Survived~., data = train_set, 
                 method = "knn", tuneGrid =data.frame(k = seq(3, 51, 2)), 
                 trControl = trainControl(method = "cv", number = 10, p = 0.9))
fit$bestTune #find the best k
y_hat_knn <- predict(fit_knn, newdata = test_set) %>% factor()
confusionMatrix(y_hat_knn, test_set$Survived)$overall["Accuracy"]

#classification tree model (rpart)
set.seed(10, sample.kind = "Rounding")
fit_rpart <- train(Survived~., data = train_set, 
                 method = "rpart", tuneGrid =data.frame(cp = seq(0, 0.05, 0.002)))
fit_rpart$bestTune
y_hat_rpart <- predict(fit_rpart, newdata = test_set) %>% factor()
confusionMatrix(y_hat_rpart, test_set$Survived)$overall["Accuracy"]
#plot the tree
plot(fit_rpart$finalModel)
text(fit_rpart$finalModel)

#Random forest model
set.seed(14)
fit_rf <- train(Survived ~., data = train_set, method = "rf", 
                tuneGrid = data.frame(mtry = seq(1:7)), 
                ntree = 100)
fit_rf$bestTune
y_hat_rf <- predict(fit_rf, newdata = test_set) %>% factor()
confusionMatrix(y_hat_rf, test_set$Survived)$overall["Accuracy"]
#the importance of the predictors
varImp(fit_rf)