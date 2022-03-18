# -*- coding: utf-8 -*-
"""
# Loading the Data

We're using the pandas library to load the data into dataframes - a tabular data structure that is perfect for this kind of work. Each of the three CSV files from Zindi is loaded into a dataframe and we take a look at the shape of the data (number of rows and columns) as well as a preview of the first 5 rows to get a feel for what we're working with.
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

# Exercise: Try this with different inputs. Any unexpected trends?

"""As you might have guessed, places that emit more light tend to be wealthier, but there is a lot of variation.

We can also look at categorical columns like 'country' or 'urban_vs_rural' and see the distribution of the target for each group:
"""

# Looking at the wealth distribution for urban vs rural
train.boxplot(by='urban_or_rural', column='Target', figsize=(12, 8))

# Exercise: which is the country with the higest average wealth_index according to this dataset?

"""Again, not unexpected. Rural areas tend to be less wealthy than urban areas.

Now the scary question: do we have missing data to deal with?
"""

train.isna().sum() # Hooray - no missing data!

"""See what other trends you can uncover - we have only scratched the surface here. """

# Exercise: explore the data further

"""# Modelling

We've had a look at our data and it looks good! Let's see if we can create a model to predict the Target given some of our inputs. To start with we will use only the numeric columns, so that we can fit a model right away. 
"""

in_cols = list(train.columns[4:-1])
print('Input columns:', in_cols)

"""To evaluate our model, we need to keep some data separate. We will split out data into X (inputs) and y (output) and then further split into train and test sets with the following code:"""

from sklearn.model_selection import train_test_split

X, y = train[in_cols], train['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58) # Random state keeps the split consistent
print(X_train.shape, X_test.shape)

"""We now have a nice test set of ~4200 rows. We will train our model and then use this test set to calculate our score.

What is the score above? The default for regression models is the R^2 score, a measure of how well the mode does at predicting the target. 0.69 is pretty good - let's plot the predictions vs the actual values and see how close it looks to a straight line:
"""

from matplotlib import pyplot as plt
plt.scatter(y_test, model.predict(X_test), alpha=0.3)

#Importing models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor 
#from sklearn.svm import SVR
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor


#Creating model
rnfreg_model = RandomForestRegressor()
linearreg_model = LinearRegression()
knnreg_model = KNeighborsRegressor()
dtreg_model = DecisionTreeRegressor(random_state=0)
#svrreg_model = SVR(kernel='linear')
xgbreg_model = XGBRegressor()
#ctb_model = CatBoostRegressor()

#training model
rnfreg_model.fit(X_train, y_train) 
linearreg_model.fit(X_train, y_train)
knnreg_model.fit(X_train, y_train)
dtreg_model.fit(X_train, y_train)
#svrreg_model.fit(X_train, y_train)
xgbreg_model.fit(X_train, y_train) 
#rnfreg_model.fit(X_train, y_train)

# Show a score
model.score(X_test, y_test)

"""This looks great - most predictions are nice and close to the true value! But we still don't have a way to link this to the leaderboard score on Zindi. Let's remedy that by calculating the Root Mean Squared Error, the same metric Zindi uses. """

from sklearn.metrics import mean_squared_error

# The `squared=False` bit tells this function to return the ROOT mean squared error
print('RandomForestRegressor: ', mean_squared_error(y_test, rnfreg_model.predict(X_test), squared=False))
print('LinearRegression: ', mean_squared_error(y_test, linearreg_model.predict(X_test), squared=False))
print('KNeighborsRegressor: ', mean_squared_error(y_test, knnreg_model.predict(X_test), squared=False))
print('DecisionTreeRegressor: ', mean_squared_error(y_test, dtreg_model.predict(X_test), squared=False))
#print('SVR: ', mean_squared_error(y_test, svrreg_model.predict(X_test), squared=False))
print('XGBRegressor: ', mean_squared_error(y_test, xgbreg_model.predict(X_test), squared=False))
#print('RandomForestRegressor: ', mean_squared_error(y_test, rnfreg_model.predict(X_test), squared=False))

"""Great stuff. Let's make a submission and then move on to looking for ways to improve.

We now have our predictions in the right format to submit. The following line saves this to a file that you can then upload to get a score:

# Getting Better

You might have noticed that your score on Zindi wasn't as good as the one you got above. This is because the test set comes from different countries to the train set. When we did a random split, we ended up with our local train and test both coming from the same countries - and it's easier for a model to extrapolate within countries than it is for it to make predictions for a new location. 

So our first step might be to make a scoring function that splits the data according to country, and measures the model performance on unseen countries. Try it and share your testing methods in the discussions. And look at the following questions:
- Does your score drop when you score your model on countries it wasn't trained with?
- Does the new score more accurately match the leaderboard score?
- Are any countries particularly 'hard' to make predictions in?
"""

# You code for a enw model evaluation method here

"""Knowing how well our model is doing is useful, but however you measure that we also need ways to improve this performance! There are a few ways to do this:

- Feed the model better data. How? Feature engineering! If we can add meaningful features the model will have more data to work with.
- Tune your models. We used the default parameters - perhaps we can tweak some hyperparameters to make our models better
- Try fancier models. Perhaps XGBoost or a neural network is better than Random Forest at this task

Let's do a little of each. First up, let's create a numeric feature that encodes the 'urban_or_rural' column as something the model can use:
"""

# Turning a categorical column into a numeric feature
train['is_urban'] = (train['urban_or_rural'] == 'U').astype(int)
test['is_urban'] = (test['urban_or_rural'] == 'U').astype(int)
train.head()

"""
Note that whenever we add features to train, *we also need to add them to test* otherwise we won't be able to make our predictions.

With this extra feature, we can fit a new model:"""

in_cols.append('is_urban') # Adding the new features to our list of input columns

# Replace this with your chosen method for evaluating a model:
X, y = train[in_cols], train['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)

pip install catboost

from sklearn.svm import SVR
from catboost import CatBoostRegressor

#Creating model2
#rnfreg2_model = RandomForestRegressor()
#linearreg2_model = LinearRegression()
#knnreg2_model = KNeighborsRegressor()
#dtreg2_model = DecisionTreeRegressor()
svrreg_model = SVR()
#xgbreg2_model = XGBRegressor()
#ctb_model = CatBoostRegressor()

#training model
#rnfreg2_model.fit(X_train, y_train) 
#linearreg2_model.fit(X_train, y_train)
#knnreg2_model.fit(X_train, y_train)
#dtreg2_model.fit(X_train, y_train)
svrreg_model.fit(X_train, y_train)
#xgbreg2_model.fit(X_train, y_train) 
#ctb_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

# The `squared=False` bit tells this function to return the ROOT mean squared error
#print('RandomForestRegressor: ', mean_squared_error(y_test, rnfreg2_model.predict(X_test), squared=False))
#print('LinearRegression: ', mean_squared_error(y_test, linearreg2_model.predict(X_test), squared=False))
#print('KNeighborsRegressor: ', mean_squared_error(y_test, knnreg2_model.predict(X_test), squared=False))
#print('DecisionTreeRegressor: ', mean_squared_error(y_test, dtreg2_model.predict(X_test), squared=False))
print('SVR: ', mean_squared_error(y_test, svrreg_model.predict(X_test), squared=False))
#print('XGBRegressor: ', mean_squared_error(y_test, xgbreg2_model.predict(X_test), squared=False))
#print('CatBoostRegressor: ', mean_squared_error(y_test, ctb_model.predict(X_test), squared=False))

"""Did your score improve?

Next, let's tune our model by adjusting the maximum depth. This is one of many hyperparameters that can be tweaked on a Random Forest model. Here I just try a few randomly chosen values, but you could also use a grid search to try values more methodically.
"""

for max_depth in [3, 5, 8, 10, 14, 18]:
    model = RandomForestRegressor()
    # Again, you van use a better method to evaluate the model here...
    model.fit(X_train, y_train)
    print(max_depth, mean_squared_error(y_test, model.predict(X_test), squared=False))

# Fit a model on the whole training set, using our best parameters
rf = RandomForestRegressor(max_depth=18)
rf.fit(X_train, y_train)

"""In this case, it looks like we can improve our performance by specifying a max_depth to limit model complexity.

Finally, let's try a different model out of curiosity:

Remember, you can ask questions and share ideas in the discussions. 

### GOOD LUCK!
"""

# Copying our desired predictions into the submission dataframe - make sure the rows are in the same order!
ss['Target'] = svrreg_model.predict(test[in_cols]) 
ss.head()

ss.to_csv('svr.csv', index=False)