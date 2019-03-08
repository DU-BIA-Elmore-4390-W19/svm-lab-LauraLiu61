SVM\_Lab
================
Yao Liu
3/5/2019

1.  This question refers to Chapter 9 Problem 8 beginning on page 371 in the text.

    1.  Create a training sample that has roughly 80% of the observations. Use `set.seed(19823)`.

``` r
df <- tbl_df(OJ)
attach(df)
set.seed(19823)
inTraining <- createDataPartition(Purchase, p = .5, list = F )
training <- df[inTraining, ]
testing  <- df[-inTraining, ]
```

    b. Use the `kernlab` package to fit a support vector classifier to the 

training data using `C = 0.01`.

``` r
purchase_svm_m1 <- ksvm(Purchase ~ ., data = training,
                     type = "C-svc", kernel = 'vanilladot', 
                     C = 0.01, prob.model = T)  
```

    ##  Setting default kernel parameters

    c. Compute the confusion matrix for the training data. Report the overall 

error rates, sensitivity, and specificity.

``` r
confusionMatrix(table(predict(purchase_svm_m1, newdata = testing),testing$Purchase),positive = "MM")
```

    ## Confusion Matrix and Statistics
    ## 
    ##     
    ##       CH  MM
    ##   CH 300  63
    ##   MM  26 145
    ##                                         
    ##                Accuracy : 0.8333        
    ##                  95% CI : (0.799, 0.864)
    ##     No Information Rate : 0.6105        
    ##     P-Value [Acc > NIR] : < 2.2e-16     
    ##                                         
    ##                   Kappa : 0.6379        
    ##  Mcnemar's Test P-Value : 0.0001356     
    ##                                         
    ##             Sensitivity : 0.6971        
    ##             Specificity : 0.9202        
    ##          Pos Pred Value : 0.8480        
    ##          Neg Pred Value : 0.8264        
    ##              Prevalence : 0.3895        
    ##          Detection Rate : 0.2715        
    ##    Detection Prevalence : 0.3202        
    ##       Balanced Accuracy : 0.8087        
    ##                                         
    ##        'Positive' Class : MM            
    ## 

    d. Construct the ROC curve. 

``` r
fits_svc <- predict(purchase_svm_m1, newdata = training, type = "probabilities")
new_fits <- mutate(training, 
                   svc_probs = fits_svc[,2],
                   default = if_else(Purchase == "CH", 0, 1))          
p <- ggplot(data = new_fits,
            aes(d = default , m = svc_probs))
p + geom_roc(n.cuts = 0, col = "orange") +
  style_roc()
```

![](svm-lab_files/figure-markdown_github/unnamed-chunk-4-1.png)

    e. Use the `train` function from the `caret` package to find an optimal cost

parameter (`C`) in the range 0.01 to 10. Use `seq(0.01, 10, len = 20)`.

``` r
fit_control <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 3)
purchase_train <- train(Purchase ~ .,
                     data = training,
                     method = "svmLinear",     #apply hyperplane here 
                     trControl = fit_control,
                     tuneGrid = data.frame(C = 0.01:10)) #add the buffer area here. 10 means the larger area of buffer 
purchase_train 
```

    ## Support Vector Machines with Linear Kernel 
    ## 
    ## 536 samples
    ##  17 predictor
    ##   2 classes: 'CH', 'MM' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 482, 482, 483, 483, 482, 482, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   C     Accuracy   Kappa    
    ##   0.01  0.8259214  0.6272805
    ##   1.01  0.8258977  0.6304291
    ##   2.01  0.8246748  0.6281250
    ##   3.01  0.8259214  0.6306227
    ##   4.01  0.8246636  0.6284906
    ##   5.01  0.8240579  0.6271953
    ##   6.01  0.8240696  0.6269202
    ##   7.01  0.8234290  0.6251337
    ##   8.01  0.8234174  0.6253875
    ##   9.01  0.8221591  0.6228205
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was C = 0.01.

    f. Compute the training and test classification error.
    g. Repeat (b) - (d) using an SVM with a polynomial kernel with degree 2. 
    h. Which method would you choose?
    i. Repeat (b) - (d) using an SVM with a radial basis kernel. Train it. 
    j. Using the best models from LDA, SVC, SVM (poly), and SVM (radial), 
    compute the test error. 
    k. Which method would you choose?

1.  Train one of the SVM models using a single core, 2 cores, and 4 cores. Compare the speedup (if any).
2.  You might want to look at `rbenchmark` or `microbenchmark` packages for timing.
