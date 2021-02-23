# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:27:18 2020
FINS3648 PYTHON ASSIGNMENT
@author: James To
@zID: z5113921
"""

########################## FINS3648 PYTHON ASSIGNMENT ########################
############################### by James To ##################################

########################### EXECUTIVE SUMMARY ################################

'''
Given the task of finding the effect of living area on house prices, the team
found that there was indeed a positive effect of living area on the sale price.
This discovery was achieved by performing data analysis and using two types
of predictive models:
    - Linear Regression
    - Random Forest
    
In doing so, we were able to build models and improve them by interpreting 
results. Furthermore, we analysed the models with more variables to see 
improvements in predictive power and interpretability. Though both models were
decent, we recommended the use of Generalized Linear Models for future
analysis.
'''

############################## INTRODUCTION ##################################

'''
My team was provided a dataset that pertains property data of an unknown 
region. We were asked to investigate the relationship between the living 
area and the sale price of houses. This was done by first performing 
Exploratory Data Analysis (EDA) and then fitting predictive models to the 
variables. Finally, we compared the models and came up with a recommendation.
'''

############################ START OF CODE ###################################

'''
We first imported all the libraries required for Exploratory Data Analysis and 
Predictive Modelling. These packages helped us visualise and create 
algorithms for our models. For example, "matplotlib.plot" was used for plotting
graphs, which will be used for visual explanations.
'''

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.ensemble import RandomForestRegressor as RandomForest
import seaborn as sns
import statsmodels.api as sm

'''
We then needed to import our dataset and we simply named it 'df'. We use the 
"Pandas" package to import with "read.csv", which loaded the CSV files into 
Python.

N/W: Prior to this step you must set the folder with the data set inside it as
your "working directory". This can be done with the following lines of code:
    import os
    os.chdir("*Insert Folder Path to Dataset*")
'''

# Load Property Dataset
df = pd.read_csv("datasets_houses.csv")

'''
After importing the data we took our variables of interest out of the dataset. 
We took the data from column "GrLivArea" and called it "xvar" and we took the 
column "SalePrice" and called it "yvar". When we made our predictive models, 
we set our "xvar" as the predictor, whichmeans its the variable we are trying 
to predict with, and we set our "yvar" as the response, which is the variable 
we are trying to predict.
'''

# Defining predictor and response variables
xvar = df[['GrLivArea']].values
yvar = df['SalePrice'].values

################### EXPLORATORY DATA ANALYSIS (EDA) ##########################

'''
We then did some Exploratory Data Analysis (EDA) on the original dataset. 
The pupose of EDA is to find any hidden details about the data that can help us 
build our models or any potential issues with the data. 

Below we plotted a scatterplot of living area vs sale price.
'''

# Scatterplot of Living Area and Sales Price
plt.plot(xvar, yvar, 'ro', color = "red")
plt.xlabel('Living Area')
plt.ylabel('Sale Price')
plt.title('Living Area vs Sale Price')
plt.show()

'''
The points are in a "cone shape", which shows us that there is more and 
more dispersion of values as both variables get larger. This meant our current 
variables have a nonlinear relationship. This could be a problem with linear 
models, and one potential fix by applying the logarithmic function to both 
variables. This is visualised by the code below:
'''

# Looking at the log relationship 
plt.plot(np.log(xvar), np.log(yvar), "ro", color = "black")
plt.show()

'''
As shown above the plot is less "cone" shaped and has less dispersion. The log
function decreases the variab;e at an increasing rate as the values get larger.
This could be a useful transformation for linear regression models.

Another concern of the data was the "YrSold" variable, which contains the year 
2008. As you may know, 2008 was the year of the GFC, where the housing market
collapsed. To find if there is any significant impact on the data quality due 
to this year, the team examined the boxplot comparing the years present in the 
dataset.
'''

# Create Boxplot for the YrSold variable against Price
sns.boxplot(x='YrSold', y='SalePrice', data=df)
plt.show()

'''
From the boxplot we see no significant changes in average house prices in 2008,
hence the team decided to leave observations from 2008 in. This could 
be because the location in which data originates from a place not directly
affected by housing market crash - like a very small city in the US.
'''

################################ MODELLING ###################################

'''
This is where we begin our predictive modelling. The goal is to discover if we
can make predictions of house prices based on living area. The models will be
judged by two performance measures: RMSE and R-Squared. The definitions are as
follows:
    - RMSE: Root Mean Squared Error is the root sum of the average error our 
    predictions were away from the real values. The lower, the better.
    - R-Squared: Tells us what percent of our predictor variables 
    explain the response variable. The closer to 1 (or 100%) the better.

We then define our first function "fit_plot", which we use for visualising our 
later linear regression models. We will use this function to essentially draw 
our prediction on our data.
'''

# Define function that draws the line of best fit
def fit_plot(xvar, yvar, model):
    plt.scatter(xvar, yvar, c='red')
    plt.plot(xvar, model.predict(xvar), color='blue', linewidth=1.5)
    return

'''
Firstly, we must split the data into training and testing sets. The training 
set is how predictive model will "learn" our data and the testing set is used 
to test the performance of our model on new data. 

A general rule of thumb is to partition the data in a 70/30 split,
where the training set has 70% of the data and the testing set has 30% of the 
data. Furthermore, we split the data into four types of data:
    - Predictor (GrLivArea) Training Set (x_train)
    - Predictor (GrLivArea) Testing Set (x_test)
    - Response (SalePrice) Training Set (y_train)
    - Response (SalePrice) Testing Set (y_test)
'''

# Data splitting
x_train, x_test, y_train, y_test = train_test_split(
    xvar, yvar, test_size=0.3, random_state=1)

###################### SIMPLE LINEAR REGRESSION ##############################

'''
The first model chosen was a simple linear regression. This is a good base 
model to build upon as well as offers high levels of interpretability.
A simple linear regression is essentially a "line of best fit" between our 
predictor and response variable.
'''

# Define simple linear regression model function
lr = LinearRegression()

# Fit the model with training dataset for it to "learn"
lr.fit(x_train, y_train)

# Predict simple linear regression results for training and testing dataset
y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

'''
The coefficients tell us what effect our predictor variable has on the 
response. For example, say we were simply comparing x and y, if the coefficient
"x" is 2, then for every unit of "x", our response increases by 2.
'''

# Print results of the model
print('GrLivArea Coefficient: %.3f' % lr.coef_[0])
print('Intercept: %.3f' % lr.intercept_)

'''
Our coefficient for "GrLivArea" is 100.273. This means, for every unit increase
living area, the sale price increases by $100.273.
'''

# Generate and print RMSE and R-Squared values
print('Simple Linear Regression RMSE train: %.3f, test: %.3f' % (
        MSE(y_train, y_train_pred)**(1/2),
        MSE(y_test, y_test_pred)**(1/2)))
print('Simple Linear Regression R^2 train: %.3f, test: %.3f' % (
        R2(y_train, y_train_pred),
        R2(y_test, y_test_pred)))

'''
The performance of our model was as follows:
    - Simple Linear Regression RMSE train: 55953.432, test: 56605.876
    - Simple Linear Regression R^2 train: 0.474, test: 0.551

The RMSE will be used for comparison with other models. However, we can examine
the R-Squared value and see that our predictor explained 47.4% and 55.1% of our
response in the train and test set respectively. Though this is perfect, 
considering that this is only ONE predictor variable, this means it does a good
job in predicting. 

The next lines of code uses the "fit_plot" function we created earlier. This
helps us visualise the line of best fit of our model.
'''

# Plot results in combined views
fit_plot(xvar, yvar, lr)
plt.xlabel('Size of Living Area [GrLivArea]')
plt.ylabel('Sale Price [SalePrice]')
plt.show()

'''
Afterwards, we looked at how the "residuals" were mapped against our model. 
"Residuals" are defined as our predicted value subtracted by our actual values,
essentially our prediction errors. For a good linear regression model, we want
to see the points scattered around 0 - portrayed by the red line - for all 
predicted (fitted) values.
'''

# Further diagnostics: Residual Analysis
plt.scatter(y_test_pred,  y_test_pred - y_test, 
           color = "grey", marker='s', label='Test data')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.axhline(color = 'red')
plt.title("Fitted vs Residual")
plt.show()

'''
The residuals spread in a somewhat "cone" shape pattern, indicating that the 
model does well for predicting small values of Sale Price, but does poorly for
larger values. This is known as heteroskedasticity, and is undesirable for 
linear models as it can over inflate the coefficients of our predictors.
'''

###################### LOG LINEAR REGRESSION #################################

''' 
We then tried to build a model based on the log-transformed relationship to see 
if we get better results, and combat the heteroskedasticity.
'''

# Define ML based linear regression parameters
loglr = LinearRegression()

# Fit SLR model with training dataset
loglr.fit(np.log(x_train), np.log(y_train))

# Predict SLR model results for training and testing dataset
logy_train_pred = loglr.predict(np.log(x_train))
logy_test_pred = loglr.predict(np.log(x_test))

# Print results of the model
print('Log Transformed LR Coefficient: %.3f' % loglr.coef_[0])
print('Log Transformed LR Intercept: %.3f' % loglr.intercept_)

# Generate and print SLR MSE and R^2 values
print('Log Transformed Linear Regression MSE train: %.3f, test: %.3f' % (
        MSE(np.log(y_train), logy_train_pred)**(1/2),
        MSE(np.log(y_test), logy_test_pred)**(1/2)))
print('Log Transformed Linear Regression R^2 train: %.3f, test: %.3f' % (
        R2(np.log(y_train), logy_train_pred),
        R2(np.log(y_test), logy_test_pred)))

# Plot results in combined views
fit_plot(np.log(xvar), np.log(yvar), loglr)
plt.xlabel('Log Size of Living Area [GrLivArea]')
plt.ylabel('Log Sale Price [SalePrice]')
plt.show()

'''
Our log-transformed model improved the R-Squared of our train set to 53% from 
47.4%, but decreased the test R-Squared to 53.4% from 55.1%. Since we are
primarily concerned with predicting - meaning the test set - the previous model
performed better.

Note: We cannot compare the RMSE of the two models because they are based on
different values (this one is in log scale).
'''

# Further diagnostics: Residual Analysis
plt.scatter(logy_test_pred,  logy_test_pred - np.log(y_test), 
           color = "grey", marker='s', label='Test data')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.axhline(color = 'red')
plt.title("Fitted vs Residual")
plt.show()

'''
However, looking at the residuals we can see that the points are much more 
spread out around the red line, and there isn't heteroskedasticity. This means
there is potential in a log-based model, which is discussed further in the 
'Recommendations' section of our report.

To improve our predictions, we looked to use more variables.
'''

##################### MULTIPLE LINEAR REGRESSION #############################

'''
The new variables selected have been selected based on logic and domain 
knowledge. Furthermore, the variables didn't have direct relation with living
are such as number of rooms (TotRmsAbvGrd) or size of lot (LotArea), and this
is to ensure the correlated variables do not overinflate their coefficients. 
The variables chosen were as follows:    
    - OverallQual: Quality rating of the property
    - OverallCond: Condition rating of the property
    - YearBuilt: The year the property was built
    - MiscVal: Any extra miscellaneous dollar value of the property
    
We then redefine our predictor variable to "xvar2", which contains all these
variables. In addition, we must split the original data into train and test 
sets with the new predictors. We then fitted the model and analysed the 
coefficients.
'''
# Set new predictor variable
xvar2 = pd.DataFrame(df[['GrLivArea','OverallQual','OverallCond','YearBuilt',
            'MiscVal']].values)

# Split the data with new predictors
x_train2, x_test2, y_train, y_test = train_test_split(
    xvar2, yvar, test_size=0.3, random_state=1)

# Define ML based linear regression parameters
lr2 = LinearRegression()

# Fit SLR model with training dataset
lr2.fit(x_train2, y_train)

# Predict SLR model results for training and testing dataset
y_train_pred2 = lr2.predict(x_train2)
y_test_pred2 = lr2.predict(x_test2)

'''
After successfully fitting the model, we analysed the coefficients.
'''

# Print results of the model
print('GrLivArea Coefficient: %.3f' % lr2.coef_[0])
print('OverallQual Coefficient: %.3f' % lr2.coef_[1])
print('OverallCond Coefficient: %.3f' % lr2.coef_[2])
print('YearBuilt Coefficient: %.3f' % lr2.coef_[3])
print('MiscVal Coefficient: %.3f' % lr2.coef_[4])
print('Intercept: %.3f' % lr2.intercept_)

'''
The coefficients were as follows:
    - GrLivArea Coefficient: 59.413
    - OverallQual Coefficient: 23373.373
    - OverallCond Coefficient: 6507.359
    - YearBuilt Coefficient: 632.668
    - MiscVal Coefficient: -0.698
    
We can see that an increase in quality had the biggest effect followed by the 
condition of the property. How new the house was also positively effected the 
price. Unexpectedly, miscelanneous value of the property slightly decreased the
sale price of the preoperty. Finally, a unit increase in living area didn't 
have a large effect (Coefficient of 59.413). However, it is important to note 
the size of the values DOES effect the coefficient, as OverallQual ranges from 
1 - 10, whereas living area value ranges from 334 - 5,642, the coefficients 
would need to be inflated for the smaller range.
'''

# Generate and Print Performance Measures
print('Multiple Linear Regression RMSE train: %.3f, test: %.3f' % (
        (MSE(y_train, y_train_pred2)**(1/2)),
        MSE(y_test, y_test_pred2)**(1/2)))
print('Multiple Linear Regression R^2 train: %.3f, test: %.3f' % (
        R2(y_train, y_train_pred2),
        R2(y_test, y_test_pred2)))

'''
The performance measures were as follows:
    - Multiple Linear Regression RMSE train: 41253.184, test: 38266.479
    - Multiple Linear Regression R^2 train: 0.714, test: 0.795
    
We can compare the RMSE values with our simple linear regression. We can see
improvement as the Multiple linear Regression's RMSE is lower. This is to be 
expected as more predictors improves our prediction. Furthermore, it showcases
that the variable "GrLivArea" alone isn't sufficient in predicting "SalePrice"
by itself. Additionally, the R-Sqaured is closer to 1, meaning these predictors
are explaining the response variable better than the previous models. 
'''

# Further diagnostics: Residual Analysis
plt.scatter(y_test_pred2,  y_test_pred2 - y_test, 
           color = "grey", marker='s', label='Test data')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.axhline(color = 'red')
plt.title("Fitted vs Residual")
plt.show()

'''
The scatterplot of the residuals shows that the predictions are closer to the
actuals and that they are better across a majority of the values. However, as
the values become higher, the residuals do get larger. 

The linear models have been great at helping us interpret the relationship 
between living area and house price, but since the response is nonlinear, it 
hasn't had the greatest accuracy. To improve on predictive power, Random Forest
model was used with the same predictors.
'''

######################## RANDOM FOREST #######################################

'''
Random Forest is known for its predictive power. The model works by making
a bunch of decision trees, and to understand the model we must understand
decision trees.

A decision tree essentially looks at each variable and asks a question. For 
example, "What value of living area would I see a substantial jump in price?" 
After asking the question, the model will look for the best value of living 
area that would substantially increase price. For example it may choose when
'Living Area > 4000'. From there it will ask another question about a new 
variable, such as "What quality rating would price increase?" Each new question
represents a 'split' in the decision tree. Each split makes two new questions, 
for when it is greater than a certain value, and less than a certain value.
The tree would build until all the predictors are exhausted. In addition, 
the predictor is picked based on if they can contribute more to the model's 
accuracy - basically picking the predictor that reduces error in prediction.

Now, a Random Forest is a bunch of different decision trees, and we take the 
weighted average of all the decision trees to make our prediction. However, 
randomness is introduced at each split of the tree, such that only a few 
predictors out of all the predictors are even considered to be asked a
question.

Below we fit the model and use it to predict our response in train and test 
set.
'''

# Create the random forest using a randomisation of 1000 observations
forest = RandomForest(n_estimators=1000,
                               criterion='mse',
                               random_state=1,
                               n_jobs=-1)

# Fit training data to model
forest.fit(x_train2, y_train)
# Train the model on training data and predict test data
rfy_train_pred = forest.predict(x_train2)
rfy_test_pred = forest.predict(x_test2)

# Look at the performance measures in terms of RMSE and R-Squared
print('RF RMSE train: %.3f, test: %.3f' % (
        MSE(y_train, rfy_train_pred)**(1/2),
        MSE(y_test, rfy_test_pred)**(1/2)))
print('RF R^2 train: %.3f, test: %.3f' % (
        R2(y_train, rfy_train_pred),
        R2(y_test, rfy_test_pred)))

'''
The performance measures were as follows:
    - RF RMSE train: 14048.863, test: 31480.039
    - RF R^2 train: 0.967, test: 0.861

This is the smallest RMSE we have achieved, proving the predictive power of 
Random Forest. Furthermore, both the R-Squared values improved such that for
the train set the predicotrs explain 96.7% of the response, and the test set 
they explain 86.1% of the response variable.

We can then see which variables were considered the most important in the
model for predicting Sale Price. This is known as Variable Importance.
'''

# See which variables the model saw as most important to predict sales price
variable_importance = pd.DataFrame(forest.feature_importances_, 
                                   index = x_train2.columns,
                                   columns=['importance']).sort_values(
                                       'importance', ascending=False)
                    
print(variable_importance)      

'''
According to random forest, 'OverallQual' was the most important variable for
predicting Sales Price. This was followed by 'GrLivArea', indicating the living
area was indeed more important than the other three predictors used.
'''                                 

########################### MODEL COMPARISONS ################################

'''
The two models I am comparing is the Multiple Linear Regression (MLR) model and 
the Random Forest model. This is because they are both different models with 
the same predictors, and had the best results of the models tested. However, 
it's important to note that the MLR was built upon prior linear regression
models.

First, we analysed the models from the perspective of accuracy. Random Forest 
had a lower test RMSE (31,480) than the MLR (38,266). Furthermore, it had
better R-Squared scores, meaning it was better at using the variables to 
explain the response variable. The better results of Random Forest can be
attributed to its better predictive power, as it can predict complex data.

Next, we looked at the interpretability of the models. With MLR, we were able
to examine the coefficients, and analyse the residuals. This helped us paint 
better picture on ways to improve the model and why the model was performing
poorly, such as the heteroskedasticity. With Random Forest, we were able to see
the variable importance of the model, and we saw that living area was the
second most important variable out of the predictors chosen. In general, MLR
gave us more interpretability. 

By comparing the two, we see there is a trade off between accuracy and 
interpretability. However, we are most concerned with how living area effects
the price, hence we are looking for more interpretable models. With linear
regression and its variations, we saw that living area did have a positive
effect on the price. However, both models also showcased that the addition
of more predictors would be more effective at forecasting prices.

If we had to pick a model between the two, it would be the MLR for
interpretability. However, we would like to suggest another model that would
provide both predictive power and interpretability.
'''

############################# RECOMMENDATIONS ################################

'''
The team and I found that Generalized Linear Models (GLM) would be a better 
model for this project as it offers both accuracy and interpretability. The 
reason for this is that it can handle the skewed data of the response variable.
This is skewed data is shown by the histogram below:
'''

plt.hist(yvar, bins = 15, color = "darkgreen")
plt.title("Histogram of Sale Price")
plt.show()

'''
Given that the price is continuous and nonlinear, the response can be optimally
log-transformed with GLM using a "Log Link Function". Furthermore,
GLMs can have "feature selection" and we can use the vast variety of predictor
variables. We will spare the technical details of this model, but we believe
it would be a good choice for this sort of analysis and prediction.

Another recommendations would be to find more observations for better analysis
and predictions. With only 1,460 observations, it is difficult to make any
absolute conclusive statements on the data.
'''

################################ CONCLUSION ##################################

'''
After going through the models, we found the the living area did have a
positive effect on house price. This was confirmed by using two types of
models, linear regression models and Random Forest. Both had decent accuracy,
with Random Forest being the best of the two, but linear regression offered 
interpretability. Ultimately, we recommended another model that can offer both,
and that was the Generalized Linear Model.
'''

############################## APPENDIX ######################################
'''
The following code is the MLR model using another function. This helped us see 
the "p-values", which tells us if a variable was statistically significant or 
not. The variable "GrLivArea" was indeed significant, as seen by a 
p-value < 0.05.
'''
xvar3 = sm.add_constant(xvar2)
est = sm.OLS(yvar, xvar2)
est2 = est.fit()
print(est2.summary())