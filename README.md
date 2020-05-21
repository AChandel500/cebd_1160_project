# CEBD 1160 Project - Assessment of Feature Selection methods on Prediction Accuracy
Submitted to CCE for CEBD1160 Winter 2020 Project Requirement 
Submitted by | Date
------------ | -------------
Amit Chandel | 28th March 2020
### Resources
* Python script for assessment: Mean_Value_Prediction.py
* Results figure/saved file: figures/
* Project Dockerfile: Dockerfile
* runtime-instructions in a file named RUNME.md

## Research Question
For selected supervised learning algorithms, do different feature selection methods impact prediction accuracy?

### Abstract

Boston housing data, made available at https://github.com/cce-bigdataintro-1160/winter2020-code/tree/master/3-python-homework/boston among 
other sources, contains features varying from pupil-teacher ratios to pollution levels that relate in varying degree to the median value of 
homes in the Boston area.

In order to answer the question of whether different feature selection approaches affect the accuracy of selected supervised regression
learning models, three methods were used to select the Boston data features provided to the models for median home value prediction. 

The accuracy of the models was then assessed using the RMSE (root mean square error) regression metric for each feature selection method, 
resulting in the confirmation that model accuracy in prediction can be positively or negatively impacted by the manner in which 
features are chosen.

### Introduction

Compiled by the United States Census Bureau, the Boston housing data used in this project can be characterized by fourteen features of numeric
data as in the list below:


    CRIM - per capita crime rate by town
    ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS - proportion of non-retail business acres per town.
    CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    NOX - nitric oxides concentration (parts per 10 million)
    RM - average number of rooms per dwelling
    AGE - proportion of owner-occupied units built prior to 1940
    DIS - weighted distances to five Boston employment centres
    RAD - index of accessibility to radial highways
    TAX - full-value property-tax rate per $10,000
    PTRATIO - pupil-teacher ratio by town
    B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT - % lower status of the population
    MEDV - Median value of owner-occupied homes in $1000's
  
The target for the models used is the median home value or MEDV.

### Methods

Six supervised machine learning models belonging to four categories were selected to predict the median value of homes in Boston:

Model type    | Name
------------  | -------------
Linear        | LinearRegression, ElasticNet, Lasso
SVM           | SVR
Neighbors     | KNeighborsRegressor
Ensemble      | GradientBoostingRegressor

Models were selected based simply on whether the data satisfies the requirements to use them, in terms of sample volume and dimensionality (feature count).

In order to select the features for these models to perform predictions, the approaches below were used:

* No criteria - all features are selected
  * Selecting all features is the simplest approach requiring the least investment to execute.
* Filter method
  * This method has a minimal computational requirement.  A correlation table between all features is assessed for features that relate highly with MEDV.  From the features selected, features that have dependencies between each other and that have the weakest correlation to MEDV are dropped.
* Embedded method
  * Relatively heavy computational requirement depending on dimensionality, as the method iteratively penalizes feature coefficients to bring their values to zero.  Features with a zero coefficient are not selected.
  * Lasso regularization (L1 regularization) is at the core of the embedded method.  The mathematical function which defines Lasso regression permits feature coefficients of the estimator function to be exactly zero (as opposed to ridge regression for example, which is very similar to Lasso) allowing for features that can be discarded to be selected before supplying data to machine learning models.
  
### Results
#### RMSE Values for Feature Selection Methods

The figure below illustrates that the no criteria method allowed for the most accurate prediction, with an RMSE value of 2.829 as the lowest score using the GradientBoostingRegressor.

![Root Mean Square Error (RMSE)](https://github.com/AChandel500/cebd_1160_project/blob/master/figures/RMSE_heatmap.png)

#### Filter and Embedded Methods Data
For the embedded and filter methods, the coefficient comparison and correlation table are respectively available below.

![Root Mean Square Error (RMSE)](https://github.com/AChandel500/cebd_1160_project/blob/master/figures/embed_method_coeffs.png)

We can see above that regularization yielded a zero-valued coefficient for the AGE feature, which was subsequently dropped for median house value prediction using the embedded method.

![Root Mean Square Error (RMSE)](https://github.com/AChandel500/cebd_1160_project/blob/master/figures/whole_corr_plot.png)

From the figure above, we can see that RM, LSTAT and PTRATIO relate highly with MEDV.  

RM was dropped by validating that the correlation between RM and LSTAT was significant at -0.6138 and that RM related to MEDV with a coefficient of 0.7. 

### Discussion

The feature selection methods applied in this assessment affected prediction accuracy significantly in some cases whereas in others the observed impact was negligible or non-existent.   

The filter method produced a signifcant decrease in prediction accuracy relative to both remaining approaches, and did so across all models.

The embedded method produced RMSE results identical to that of full feature selection (no criteria) for 50% of the models, with a marginal relative increase in accuracy using the KNeighboursRegressor model and marginal relative decreases in accuracy using the GradientBoostingRegressor and SVR models.    

Further assessment using differing datasets would provide greater insight.   

### References

https://github.com/cce-bigdataintro-1160/winter2020-code/tree/master/3-python-homework/boston

https://scikit-learn.org/stable/supervised_learning.html
