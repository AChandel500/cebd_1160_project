# CEBD 1160 Project - Assessment of Feature Selection methods on Prediction Accuracy
Submitted to CCE for CEBD1160 Winter 2020 Project Requirement 
Submitted by | Date
------------ | -------------
Amit Chandel | 28th March 2020
### Resources
* Python script for assessment: Mean_Value_Prediction.py
* Results figure/saved file: figures/
* Dockerfile for your experiment: Dockerfile
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
  * This method has a minimal computational requirement.  A correlation table between all features is assessed for features that relate highly with MEDV.  From the features selected, features that have dependencies between each other and that have the highest correlation to MEDV are dropped.
* Embedded method
  * Relatively heavy computational requirement depending on dimensionality, as the method iteratively penalizes feature coefficients to bring their values to zero.  Features with a zero coefficient are not selected.

### Results
#### RMSE Values for Feature Selection Methods

The figure below illustrates that the embedded feature selection method allowed for the most accurate predictions, across all models, with an RMSE value of 8.3 x 10<sup>-15</sup> for LinearRegression as the lowest score.

![Root Mean Square Error (RMSE)](https://github.com/AChandel500/cebd_1160_project/blob/master/figures/RMSE_heatmap.png)

#### Filter and Embedded Methods Data
For the embedded and filter methods, the coefficient comparison and correlation table are respectively available below.

![Root Mean Square Error (RMSE)](https://github.com/AChandel500/cebd_1160_project/blob/master/figures/embed_method_coeffs.png)

We can see above that INDUS, NOX and CHAS have zero valued coefficients and were dropped.

![Root Mean Square Error (RMSE)](https://github.com/AChandel500/cebd_1160_project/blob/master/figures/whole_corr_plot.png)

From the figure above, we can see that RM, LSTAT and PTRATIO relate highly with MEDV.  

RM was dropped by validating that the correlation between RM and LSTAT was significant at -0.6138.

### Discussion
Using an appropriate feature selection method can significantly improve model accuracy.

Further methods could be applied to the dataset and other datasets could be used to obtain greater confirmation.

### References

https://github.com/cce-bigdataintro-1160/winter2020-code/tree/master/3-python-homework/boston

https://scikit-learn.org/stable/supervised_learning.html
