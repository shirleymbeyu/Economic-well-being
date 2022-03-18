# -*- coding: utf-8 -*-
"""
# Loading the Data
"""

import pandas as pd

train = pd.read_csv('Train.csv')
print(train.shape)
train.head()

"""In train, we have a set of inputs (like 'urban_or_rural' or 'ghsl_water_surface') and our desired output variable, 'Target'. There are 21454 rows - lots of juicy data!"""

test = pd.read_csv('Test.csv')
print(test.shape)
test.head()

"""Test looks just like train but without the 'Target' column and with fewer rows."""

ss = pd.read_csv('SampleSubmission.csv')
print(ss.shape)
ss.head()

"""The sample submission is just the ID column from test with a 'Target' column where we will put out predictions.

Now that we have the data loaded, we can start exploring.

# EDA

We will explore some trends in the data and look for any anomalies such as missing data. A few examples are done here but you can explore much further yourself and get to know the data better.

First up: let's see how an input like 'nighttime lights' relates to the target column:
"""

# Plotting the relationship between an input column and the target
train.plot(x='nighttime_lights', y='Target', kind='scatter', alpha=0.2)

"""As you might have guessed, places that emit more light tend to be wealthier, but there is a lot of variation.

We can also look at categorical columns like 'country' or 'urban_vs_rural' and see the distribution of the target for each group:
"""

# Looking at the wealth distribution for urban vs rural
train.boxplot(by='urban_or_rural', column='Target', figsize=(12, 8))

"""Again, not unexpected. Rural areas tend to be less wealthy than urban areas.

Now the scary question: do we have missing data to deal with?
"""

train.isna().sum() # Hooray - no missing data!

"""I did very little analysis due to time 😕
(I'll review)

# Modelling
"""

in_cols = list(train.columns[4:-1])
print('Input columns:', in_cols)

# Turning a categorical column into a numeric feature
train['is_urban'] = (train['urban_or_rural'] == 'U').astype(int)
test['is_urban'] = (test['urban_or_rural'] == 'U').astype(int)
train.head()

in_cols.append('is_urban') # Adding the new features to our list of input columns

from sklearn.model_selection import train_test_split

#X, y = train[in_cols], train['Target']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58) # Random state keeps the split consistent
#print(X_train.shape, X_test.shape)

# Replace this with your chosen method for evaluating a model:
X, y = train[in_cols], train['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)

pip install catboost

from sklearn import metrics

"""I'll try all the regressor models i know 🙂

**note to self**:: research on other factors behind model selection 😀
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor 
import lightgbm as lgb

#Creating model2
rnfreg2_model = RandomForestRegressor()
linearreg2_model = LinearRegression()
knnreg2_model = KNeighborsRegressor()
dtreg2_model = DecisionTreeRegressor()
svrreg_model = SVR()
xgbreg2_model = XGBRegressor()
ctb_model = CatBoostRegressor()
gbreg_model = GradientBoostingRegressor()
abreg_model = AdaBoostRegressor()
breg_model = BaggingRegressor()
lgbm = lgb.LGBMRegressor()

#training model
rnfreg2_model.fit(X_train, y_train) 
linearreg2_model.fit(X_train, y_train)
knnreg2_model.fit(X_train, y_train)
dtreg2_model.fit(X_train, y_train)
svrreg_model.fit(X_train, y_train)
xgbreg2_model.fit(X_train, y_train) 
ctb_model.fit(X_train, y_train) 
gbreg_model.fit(X_train, y_train) 
abreg_model.fit(X_train, y_train) 
breg_model.fit(X_train, y_train) 
lgbm.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

# The `squared=False` bit tells this function to return the ROOT mean squared error
print('RandomForestRegressor: ', mean_squared_error(y_test, rnfreg2_model.predict(X_test), squared=False))
print('LinearRegression: ', mean_squared_error(y_test, linearreg2_model.predict(X_test), squared=False))
print('KNeighborsRegressor: ', mean_squared_error(y_test, knnreg2_model.predict(X_test), squared=False))
print('DecisionTreeRegressor: ', mean_squared_error(y_test, dtreg2_model.predict(X_test), squared=False))
print('SVR: ', mean_squared_error(y_test, svrreg_model.predict(X_test), squared=False))
print('XGBRegressor: ', mean_squared_error(y_test, xgbreg2_model.predict(X_test), squared=False))
print('CatBoostRegressor: ', mean_squared_error(y_test, ctb_model.predict(X_test), squared=False))
print('GradientBoostingRegressor: ', mean_squared_error(y_test, gbreg_model.predict(X_test), squared=False))
print('AdaBoostRegressor: ', mean_squared_error(y_test, abreg_model.predict(X_test), squared=False))
print('BaggingRegressor: ', mean_squared_error(y_test, breg_model.predict(X_test), squared=False))
print('LGBMRegressor: ', mean_squared_error(y_test, lgbm.predict(X_test), squared=False))

"""##**model tuning**

a) random forest
"""

for max_depth in [3, 5, 8, 10, 14, 18]:
    model = RandomForestRegressor()
    # Again, you van use a better method to evaluate the model here...
    model.fit(X_train, y_train)
    print(max_depth, mean_squared_error(y_test, model.predict(X_test), squared=False))

"""b) xgb"""

from sklearn.model_selection import GridSearchCV

xgb1 = XGBRegressor()

# Define hyperparameters
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 10, 15],
              'min_child_weight': [4, 8, 12],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

# Grid search object
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_)

"""# Submission

"""

# Fit a model on the whole training set, using our best parameters
rf = RandomForestRegressor(max_depth=18)
rf.fit(X_train, y_train)

# Fit a model on the whole training set, using our best parameters
xgb = XGBRegressor(colsample_bytree=0.7,learning_rate=0.03,max_depth=5,min_child_weight=12,n_estimators=500,nthread= 4,silent=1,subsample=0.7)
xgb.fit(X_train, y_train)

# Copying our desired predictions into the submission dataframe - make sure the rows are in the same order!
ss['Target'] = gbreg_model.predict(test[in_cols]) 
ss.head()

ss.to_csv('gbreg.csv', index=False)