Business Intelligence Lab Submission Markdown
================
naive
4/10/2023

- [Student Details](#student-details)
- [Setup Chunk](#setup-chunk)
- [DATASET 1 (Splitting the dataset): PimaIndians
  Diabetes](#dataset-1-splitting-the-dataset-pimaindians-diabetes)
  - [3.b. Test the trained caret Naive Bayes model using the testing
    dataset
    —-](#3b-test-the-trained-caret-naive-bayes-model-using-the-testing-dataset--)
  - [4. View the Results —-](#4-view-the-results--)
    - [4.a. e1071 Naive Bayes model and test results using a confusion
      matrix
      —-](#4a-e1071-naive-bayes-model-and-test-results-using-a-confusion-matrix--)
    - [5.a. SVM Classifier using 5-fold cross validation with 3 reps
      —-](#5a-svm-classifier-using-5-fold-cross-validation-with-3-reps--)
    - [5.b. Test the trained SVM model using the testing dataset
      —-](#5b-test-the-trained-svm-model-using-the-testing-dataset--)
    - [5.c. View a summary of the model and view the confusion matrix
      —-](#5c-view-a-summary-of-the-model-and-view-the-confusion-matrix--)
  - [6. Classification: Naive Bayes with Leave One Out Cross
    Validation—-](#6-classification-naive-bayes-with-leave-one-out-cross-validation-)
    - [6.b. Test the trained model using the testing dataset
      ====](#6b-test-the-trained-model-using-the-testing-dataset-)
    - [6.c. View the confusion matrix
      ====](#6c-view-the-confusion-matrix-)

# Student Details

|                                                   |                                                                                                                                                                                                                                                                                                                                                             |     |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----|
| **Student ID Numbers and Names of Group Members** | *\<list one student name, class group (just the letter; A, B, or C), and ID per line, e.g., 123456 - A - John Leposo; you should be between 2 and 5 members per group\>* \| \| 1. 135575 - B - Dennis Nzioki. \| \| 2. 134645 - B - Vivean Lydiah \| \| 3. 134765 - B - Nicholas Munene \| 4. 131653- B - Terry Joan \| \| 5. 124428 - B - Eston Gichuhi \| |     |
| **GitHub Classroom Group Name**                   | *\<specify the name of the team you created on GitHub classroom\>*                                                                                                                                                                                                                                                                                          |     |
| **Course Code**                                   | BBT4206                                                                                                                                                                                                                                                                                                                                                     |     |
| **Course Name**                                   | Business Intelligence II                                                                                                                                                                                                                                                                                                                                    |     |
| **Program**                                       | Bachelor of Business Information Technology                                                                                                                                                                                                                                                                                                                 |     |
| **Semester Duration**                             | 21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023                                                                                                                                                                                                                                                                                                |     |

# Setup Chunk

We start by installing all the required packages We start by installing
all the required packages

``` r
## formatR - Required to format R code in the markdown ----
if (!is.element("formatR", installed.packages()[, 1])) {
  install.packages("formatR", dependencies = TRUE,
                   repos="https://cloud.r-project.org")
}
require("formatR")
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## klaR ----
if (require("klaR")) {
  require("klaR")
} else {
  install.packages("klaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## e1071 ----
if (require("e1071")) {
  require("e1071")
} else {
  install.packages("e1071", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## readr ----
if (require("readr")) {
  require("readr")
} else {
  install.packages("readr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## LiblineaR ----
if (require("LiblineaR")) {
  require("LiblineaR")
} else {
  install.packages("LiblineaR", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## naivebayes ----
if (require("naivebayes")) {
  require("naivebayes")
} else {
  install.packages("naivebayes", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
     install.packages("mlbench", dependencies = TRUE)  #nolint
  
}
if (!is.element("mlbench", installed.packages()[, 1])) {}
  require("mlbench") #nolint
```

------------------------------------------------------------------------

**Note:** the following “*KnitR*” options have been set as the defaults
in this markdown:  
`knitr::opts_chunk$set(echo = TRUE, warning = FALSE, eval = TRUE, collapse = FALSE, tidy.opts = list(width.cutoff = 80), tidy = TRUE)`.

More KnitR options are documented here
<https://bookdown.org/yihui/rmarkdown-cookbook/chunk-options.html> and
here <https://yihui.org/knitr/options/>.

``` r
knitr::opts_chunk$set(
    eval = TRUE,
    echo = TRUE,
    warning = FALSE,
    collapse = FALSE,
    tidy = TRUE
)
```

# DATASET 1 (Splitting the dataset): PimaIndians Diabetes

``` r
data("PimaIndiansDiabetes")
summary(PimaIndiansDiabetes)
```

    ##     pregnant         glucose         pressure         triceps     
    ##  Min.   : 0.000   Min.   :  0.0   Min.   :  0.00   Min.   : 0.00  
    ##  1st Qu.: 1.000   1st Qu.: 99.0   1st Qu.: 62.00   1st Qu.: 0.00  
    ##  Median : 3.000   Median :117.0   Median : 72.00   Median :23.00  
    ##  Mean   : 3.845   Mean   :120.9   Mean   : 69.11   Mean   :20.54  
    ##  3rd Qu.: 6.000   3rd Qu.:140.2   3rd Qu.: 80.00   3rd Qu.:32.00  
    ##  Max.   :17.000   Max.   :199.0   Max.   :122.00   Max.   :99.00  
    ##     insulin           mass          pedigree           age        diabetes 
    ##  Min.   :  0.0   Min.   : 0.00   Min.   :0.0780   Min.   :21.00   neg:500  
    ##  1st Qu.:  0.0   1st Qu.:27.30   1st Qu.:0.2437   1st Qu.:24.00   pos:268  
    ##  Median : 30.5   Median :32.00   Median :0.3725   Median :29.00            
    ##  Mean   : 79.8   Mean   :31.99   Mean   :0.4719   Mean   :33.24            
    ##  3rd Qu.:127.2   3rd Qu.:36.60   3rd Qu.:0.6262   3rd Qu.:41.00            
    ##  Max.   :846.0   Max.   :67.10   Max.   :2.4200   Max.   :81.00

``` r
str(PimaIndiansDiabetes)
```

    ## 'data.frame':    768 obs. of  9 variables:
    ##  $ pregnant: num  6 1 8 1 0 5 3 10 2 8 ...
    ##  $ glucose : num  148 85 183 89 137 116 78 115 197 125 ...
    ##  $ pressure: num  72 66 64 66 40 74 50 0 70 96 ...
    ##  $ triceps : num  35 29 0 23 35 0 32 0 45 0 ...
    ##  $ insulin : num  0 0 0 94 168 0 88 0 543 0 ...
    ##  $ mass    : num  33.6 26.6 23.3 28.1 43.1 25.6 31 35.3 30.5 0 ...
    ##  $ pedigree: num  0.627 0.351 0.672 0.167 2.288 ...
    ##  $ age     : num  50 31 32 21 33 30 26 29 53 54 ...
    ##  $ diabetes: Factor w/ 2 levels "neg","pos": 2 1 2 1 2 1 2 1 2 2 ...

\##Split the dataset:

``` r
train_index <- createDataPartition(PimaIndiansDiabetes$diabetes, p = 0.75, list = FALSE)
PimaIndiansDiabetes_train <- PimaIndiansDiabetes[train_index, ]
PimaIndiansDiabetes_test <- PimaIndiansDiabetes[-train_index, ]
```

\##Train a Naive Bayes classifier using the training dataset this can be
done by using an naive bayes function in the e1071 package or the caret
package

``` r
PimaIndiansDiabetes_model_nb_e1071 <- # nolint
  e1071::naiveBayes(diabetes ~ pregnant + glucose + pressure + triceps + insulin + mass +
                      pedigree + age,
                    data = PimaIndiansDiabetes_train)
PimaIndiansDiabetes_model_nb_caret <- # nolint
  caret::train(diabetes ~ ., data =
                 PimaIndiansDiabetes_train[, c("pregnant", "glucose", "pressure", "triceps", "insulin",
                                               "mass", "pedigree", "age", "diabetes")],
               method = "naive_bayes")
```

\##Test the trained model using the testing dataset \### 3.a. Test the
trained e1071 Naive Bayes model using the testing dataset —-

``` r
predictions_nb_e1071 <- predict(PimaIndiansDiabetes_model_nb_e1071, PimaIndiansDiabetes_test[,
    c("pregnant", "glucose", "pressure", "triceps", "insulin", "mass", "pedigree",
        "age", "diabetes")])
```

### 3.b. Test the trained caret Naive Bayes model using the testing dataset —-

``` r
predictions_nb_caret <- predict(PimaIndiansDiabetes_model_nb_caret, PimaIndiansDiabetes_test[,
    c("pregnant", "glucose", "pressure", "triceps", "insulin", "mass", "pedigree",
        "age", "diabetes")])
```

## 4. View the Results —-

### 4.a. e1071 Naive Bayes model and test results using a confusion matrix —-

``` r
print(predictions_nb_e1071)
```

    ##   [1] neg neg pos neg pos neg neg neg neg neg pos pos neg pos neg neg neg neg
    ##  [19] neg neg neg pos pos neg neg neg pos neg neg neg pos neg pos neg neg pos
    ##  [37] neg neg pos neg neg neg neg neg neg pos neg neg neg neg neg neg neg neg
    ##  [55] neg neg neg neg neg neg neg neg neg pos neg neg neg neg pos pos neg neg
    ##  [73] pos neg neg neg neg neg neg pos neg pos pos neg pos neg neg neg neg neg
    ##  [91] pos neg neg neg neg neg pos neg pos neg neg neg neg neg neg neg pos neg
    ## [109] neg neg neg pos neg neg neg neg neg pos neg neg neg neg pos pos pos neg
    ## [127] neg neg neg neg neg neg pos neg pos neg neg neg neg neg pos neg pos neg
    ## [145] pos pos neg neg neg neg pos neg neg pos neg neg neg neg neg neg neg neg
    ## [163] neg neg neg pos neg neg pos neg neg neg neg pos pos neg pos neg neg neg
    ## [181] neg neg neg neg pos neg pos pos neg pos neg neg
    ## Levels: neg pos

``` r
caret::confusionMatrix(predictions_nb_e1071, PimaIndiansDiabetes_test[, c("pregnant",
    "glucose", "pressure", "triceps", "insulin", "mass", "pedigree", "age", "diabetes")]$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg 113  32
    ##        pos  12  35
    ##                                           
    ##                Accuracy : 0.7708          
    ##                  95% CI : (0.7048, 0.8283)
    ##     No Information Rate : 0.651           
    ##     P-Value [Acc > NIR] : 0.0002216       
    ##                                           
    ##                   Kappa : 0.4581          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.0041786       
    ##                                           
    ##             Sensitivity : 0.9040          
    ##             Specificity : 0.5224          
    ##          Pos Pred Value : 0.7793          
    ##          Neg Pred Value : 0.7447          
    ##              Prevalence : 0.6510          
    ##          Detection Rate : 0.5885          
    ##    Detection Prevalence : 0.7552          
    ##       Balanced Accuracy : 0.7132          
    ##                                           
    ##        'Positive' Class : neg             
    ## 

``` r
plot(table(predictions_nb_e1071, PimaIndiansDiabetes_test[, c("pregnant", "glucose",
    "pressure", "triceps", "insulin", "mass", "pedigree", "age", "diabetes")]$diabetes))
```

![](Lab-Submission-Markdown_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->
\### 4.b. caret Naive Bayes model and test results using a confusion
matrix —-

``` r
print(PimaIndiansDiabetes_model_nb_caret)
```

    ## Naive Bayes 
    ## 
    ## 576 samples
    ##   8 predictor
    ##   2 classes: 'neg', 'pos' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 576, 576, 576, 576, 576, 576, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   usekernel  Accuracy   Kappa    
    ##   FALSE      0.7590731  0.4532743
    ##    TRUE      0.7599405  0.4586532
    ## 
    ## Tuning parameter 'laplace' was held constant at a value of 0
    ## Tuning
    ##  parameter 'adjust' was held constant at a value of 1
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were laplace = 0, usekernel = TRUE
    ##  and adjust = 1.

``` r
caret::confusionMatrix(predictions_nb_caret, PimaIndiansDiabetes_test[, c("pregnant",
    "glucose", "pressure", "triceps", "insulin", "mass", "pedigree", "age", "diabetes")]$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg 108  31
    ##        pos  17  36
    ##                                           
    ##                Accuracy : 0.75            
    ##                  95% CI : (0.6826, 0.8096)
    ##     No Information Rate : 0.651           
    ##     P-Value [Acc > NIR] : 0.002076        
    ##                                           
    ##                   Kappa : 0.4218          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.060602        
    ##                                           
    ##             Sensitivity : 0.8640          
    ##             Specificity : 0.5373          
    ##          Pos Pred Value : 0.7770          
    ##          Neg Pred Value : 0.6792          
    ##              Prevalence : 0.6510          
    ##          Detection Rate : 0.5625          
    ##    Detection Prevalence : 0.7240          
    ##       Balanced Accuracy : 0.7007          
    ##                                           
    ##        'Positive' Class : neg             
    ## 

``` r
plot(table(predictions_nb_caret, PimaIndiansDiabetes_test[, c("pregnant", "glucose",
    "pressure", "triceps", "insulin", "mass", "pedigree", "age", "diabetes")]$diabetes))
```

![](Lab-Submission-Markdown_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->
\## 5. Classification: SVM with Repeated k-fold Cross Validation —-

### 5.a. SVM Classifier using 5-fold cross validation with 3 reps —-

the support vector machine is trained with the purpose of classification
using ’diabetes variable. we will use a 5-fold cross-validation train
control with 3 reps

``` r
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

PimaIndiansDiabetes_model_svm <- caret::train(diabetes ~ ., data = PimaIndiansDiabetes_train,
    trControl = train_control, na.action = na.omit, method = "svmLinearWeights2",
    metric = "Accuracy")
```

### 5.b. Test the trained SVM model using the testing dataset —-

``` r
predictions_svm <- predict(PimaIndiansDiabetes_model_svm, PimaIndiansDiabetes_test[,
    1:9])
```

### 5.c. View a summary of the model and view the confusion matrix —-

``` r
print(PimaIndiansDiabetes_model_svm)
```

    ## L2 Regularized Linear Support Vector Machines with Class Weights 
    ## 
    ## 576 samples
    ##   8 predictor
    ##   2 classes: 'neg', 'pos' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 461, 461, 460, 461, 461, 461, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cost  Loss  weight  Accuracy   Kappa     
    ##   0.25  L1    1       0.6483058  0.09990756
    ##   0.25  L1    2       0.6191154  0.12976006
    ##   0.25  L1    3       0.5868316  0.08390882
    ##   0.25  L2    1       0.7286057  0.34928716
    ##   0.25  L2    2       0.6688106  0.33822493
    ##   0.25  L2    3       0.4912644  0.12497952
    ##   0.50  L1    1       0.6098301  0.07088747
    ##   0.50  L1    2       0.5943628  0.12753140
    ##   0.50  L1    3       0.6492154  0.13657081
    ##   0.50  L2    1       0.7332284  0.36273528
    ##   0.50  L2    2       0.6700250  0.34015914
    ##   0.50  L2    3       0.4930035  0.12712835
    ##   1.00  L1    1       0.6133383  0.07332974
    ##   1.00  L1    2       0.5944878  0.09982781
    ##   1.00  L1    3       0.5903548  0.08011460
    ##   1.00  L2    1       0.7268816  0.34634400
    ##   1.00  L2    2       0.6757871  0.35017243
    ##   1.00  L2    3       0.4930035  0.12712835
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were cost = 0.5, Loss = L2 and weight = 1.

``` r
caret::confusionMatrix(predictions_svm, PimaIndiansDiabetes_test$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg 115  39
    ##        pos  10  28
    ##                                          
    ##                Accuracy : 0.7448         
    ##                  95% CI : (0.677, 0.8048)
    ##     No Information Rate : 0.651          
    ##     P-Value [Acc > NIR] : 0.003401       
    ##                                          
    ##                   Kappa : 0.3756         
    ##                                          
    ##  Mcnemar's Test P-Value : 6.334e-05      
    ##                                          
    ##             Sensitivity : 0.9200         
    ##             Specificity : 0.4179         
    ##          Pos Pred Value : 0.7468         
    ##          Neg Pred Value : 0.7368         
    ##              Prevalence : 0.6510         
    ##          Detection Rate : 0.5990         
    ##    Detection Prevalence : 0.8021         
    ##       Balanced Accuracy : 0.6690         
    ##                                          
    ##        'Positive' Class : neg            
    ## 

## 6. Classification: Naive Bayes with Leave One Out Cross Validation—-

In Leave One Out Cross-Validation (LOOCV), a data instance is left out
and a model constructed on all other data instances in the training set.
This is repeated for all data instances. \### 6.a. Train a Naive Bayes
classifier based on an LOOCV —-

``` r
train_control <- trainControl(method = "LOOCV")

PimaIndiansDiabetes_model_nb_loocv <- caret::train(diabetes ~ ., data = PimaIndiansDiabetes_train,
    trControl = train_control, na.action = na.omit, method = "naive_bayes", metric = "Accuracy")
```

### 6.b. Test the trained model using the testing dataset ====

``` r
predictions_nb_loocv <- predict(PimaIndiansDiabetes_model_nb_loocv, PimaIndiansDiabetes_test[,
    1:9])
```

### 6.c. View the confusion matrix ====

``` r
print(PimaIndiansDiabetes_model_nb_loocv)
```

    ## Naive Bayes 
    ## 
    ## 576 samples
    ##   8 predictor
    ##   2 classes: 'neg', 'pos' 
    ## 
    ## No pre-processing
    ## Resampling: Leave-One-Out Cross-Validation 
    ## Summary of sample sizes: 575, 575, 575, 575, 575, 575, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   usekernel  Accuracy   Kappa    
    ##   FALSE      0.7534722  0.4459212
    ##    TRUE      0.7656250  0.4663153
    ## 
    ## Tuning parameter 'laplace' was held constant at a value of 0
    ## Tuning
    ##  parameter 'adjust' was held constant at a value of 1
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final values used for the model were laplace = 0, usekernel = TRUE
    ##  and adjust = 1.

``` r
caret::confusionMatrix(predictions_nb_loocv, PimaIndiansDiabetes_test$diabetes)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction neg pos
    ##        neg 108  31
    ##        pos  17  36
    ##                                           
    ##                Accuracy : 0.75            
    ##                  95% CI : (0.6826, 0.8096)
    ##     No Information Rate : 0.651           
    ##     P-Value [Acc > NIR] : 0.002076        
    ##                                           
    ##                   Kappa : 0.4218          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.060602        
    ##                                           
    ##             Sensitivity : 0.8640          
    ##             Specificity : 0.5373          
    ##          Pos Pred Value : 0.7770          
    ##          Neg Pred Value : 0.6792          
    ##              Prevalence : 0.6510          
    ##          Detection Rate : 0.5625          
    ##    Detection Prevalence : 0.7240          
    ##       Balanced Accuracy : 0.7007          
    ##                                           
    ##        'Positive' Class : neg             
    ## 
