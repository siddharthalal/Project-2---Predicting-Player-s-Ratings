# Predicting Player Ratings Using Regression in Python

---

In this project, we are going to predict the overall rating of soccer player based on their attributes such as 'crossing', 'finishing etc.

The dataset that we are going to use is from the European Soccer Database (https://www.kaggle.com/hugomathien/soccer). 

I am going to use the [Scikit-learn](http://scikit-learn.org/stable/) ML library in Python to perform regression.

---

Let’s first import some libraries that we are going to need for our analysis:

```python
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
# allow plots to appear directly in the notebook
%matplotlib inline
```

Load the .sqlite file into a Pandas data frame.

```python
cnx = sqlite3.connect('players-database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.head()
```

![dataframe](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/dataframe.png?raw=true)

### Data clean up

```python
# Drop the columns that we won't be needing for our analysis.
df.drop(['id', 'player_fifa_api_id', 'player_api_id', 'date'], axis=1, inplace=True)

#Check for null values
df.isnull().any()
```

![dataframe](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/columns-with-nan-values.png?raw=true)

836 records which have "overall_rating" as null have null values in all the columns. We can safely drop them.

```python
df = df[~df.overall_rating.isnull()]

# replace NANs in others with series means

df["volleys"].fillna(df["volleys"].mean(),inplace=True)
df["curve"].fillna(df["curve"].mean(),inplace=True)
df["agility"].fillna(df["agility"].mean(),inplace=True)
df["balance"].fillna(df["balance"].mean(),inplace=True)
df["jumping"].fillna(df["jumping"].mean(),inplace=True)
df["vision"].fillna(df["vision"].mean(),inplace=True)
df["sliding_tackle"].fillna(df["sliding_tackle"].mean(),inplace=True)
```

Clean up values in defensive_work_rate and attacking_work_rate columns
```python
df.defensive_work_rate.value_counts()
```

![defensive rate value counts](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/defensive-work-rate-value-counts-1.png?raw=true)

Converting defensive_work_rate is tricky. The acceptable values are high, medium and low, but the column has lot of other values which don't make sense. And since we don't have the metadata available for the column, the safest choice would be to drop all the rows having non-sensical values. But before we do that, lets try to make sense of the given data.

Few rows have numerical values ranging from 0-9. For them, we can assume that 0-3 means "low", 4-6 means "medium", 7-9 means high. Three other values "o", "_0" and "ormal" can be interpreted as "0", "0" and "normal" which can in turn be interpreted as "low", "low" and "medium". Rest of the rows can be dropped.

```python
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('_0','0')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('ormal','5')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('o','0')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('l0w','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('0','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('1','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('2','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('3','low')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('4','medium')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('5','medium')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('6','medium')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('7','high')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('8','high')
df['defensive_work_rate'] = df['defensive_work_rate'].str.replace('9','high')

#Choose only the rows having work rate as low, medium and high

df = df[(df.defensive_work_rate == 'medium') | (df.defensive_work_rate == 'high') | (df.defensive_work_rate == 'low')]
df.defensive_work_rate.value_counts()
```

![defensive rate value counts](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/defensive-work-rate-value-counts-2.png?raw=true)

attacking_work_rate

```python
df.attacking_work_rate.value_counts()
```

![defensive rate value counts](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/attacking-work-rate-value-counts.png?raw=true)

Converting attacking_work_rate is again tricky. The acceptable values are high, medium and low, but the column has other values which don't make much sense. And since we don't have the metadata available for this column, the safest choice would be to drop all the rows having values other than low, high and medium.

```python
# Change "norm" to "medium" and drop the rest having "None" and "Null" values.
df['attacking_work_rate'] = df['attacking_work_rate'].str.replace('norm','medium')
df = df[(df.attacking_work_rate == 'medium') | (df.attacking_work_rate == 'high') | (df.attacking_work_rate == 'low')]
```

Since we know the particular order of values in the "attacking_work_rate" & "defensive_work_rate" features, we can numerically encode them as 0 for "low", 1 for "medium" and 2 for "high".

```python
df['attacking_work_rate'] = df['attacking_work_rate'].map({'low': 0, 'medium': 1, 'high': 2}).astype(int)
df['defensive_work_rate'] = df['defensive_work_rate'].map({'low': 0, 'medium': 1, 'high': 2}).astype(int)
df['preferred_foot'] = df['preferred_foot'].map({'left': 0, 'right': 1}).astype(int)
```

### Data Exploration

Check correlation between the features and the target column.

```python
from math import ceil

fig = plt.figure(figsize=(30,20))
cols = 5
rows = ceil(float(df.shape[1]) / cols)
for i, column in enumerate(df.columns):
    axs = fig.add_subplot(rows, cols, i + 1)
    axs.set_title(column)
    df.plot(kind='scatter', x=column, y='overall_rating', ax=axs)
    plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.9, wspace=0.3)
```

![features-target-correlation](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/features-target-correlation.png?raw=true)

The "overall rating" feature doesn't seem to be linearlly correlated to all the other features except for a few like "potential". Running a linear regression model on non-linear data wouldn't give us the best results. We will test this. Let's see how the features are correlated to each other.

```python
fig = plt.figure(figsize=(12,9))
sns.heatmap(df.corr(), square=True)
plt.show()
```

![features-target-correlation](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/features-correlation-heatmap.png?raw=true)

### Build the model

```python
#The columns that we will be making predictions with.
X = df_modified[['potential', 'preferred_foot_left', 'preferred_foot_right', 'crossing', 'finishing',
       'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve',
       'free_kick_accuracy', 'long_passing', 'ball_control', 'acceleration',
       'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power',
       'jumping', 'stamina', 'strength', 'long_shots', 'aggression',
       'interceptions', 'positioning', 'vision', 'penalties', 'marking',
       'standing_tackle', 'sliding_tackle',
       'attacking_work_rate_high', 'attacking_work_rate_low',
       'attacking_work_rate_medium', 'defensive_work_rate_high',
       'defensive_work_rate_low', 'defensive_work_rate_medium']]
       
#The column that we want to predict.
y = df_modified["overall_rating"]
```

There is a strong correlation between various features which means we also have a problem of data redundancy.

We can remove highly correlated predictors from the model using Random Forest regression technique. Because they supply redundant information, removing one of the correlated factors usually doesn't drastically reduce the R-squared.

Based on initial data analysis, random forest regression seems a better technique that linear regression in this case. Let's evaluate both of them.

### Build the model

The columns that we will be making predictions with.

```python
X = df[['potential', 'preferred_foot', 'crossing', 'finishing',
       'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve',
       'free_kick_accuracy', 'long_passing', 'ball_control', 'acceleration',
       'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power',
       'jumping', 'stamina', 'strength', 'long_shots', 'aggression',
       'interceptions', 'positioning', 'vision', 'penalties', 'marking',
       'standing_tackle', 'sliding_tackle',
       'attacking_work_rate', 'defensive_work_rate']]
```

The column that we want to predict.

```python
y = df["overall_rating"]       
```

### Linear Regression

Evaluate the model by splitting into train and test sets.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lm = LinearRegression()
lm.fit(X_train, y_train)

predicted = lm.predict(X_test)
print ("Test Accuracy:", round(metrics.r2_score(y_test, predicted) * 100, 2), '%')
print ("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predicted),5))
```

Test Accuracy: 79.2 %

Mean Squared Error: 10.33402

Mean squared error seems high. Most machine learning algorithms like the features to be scaled with mean 0 and variance 1. This is called normalization which means “removing the mean and scaling to unit variance”. Lets try that and run our model again.

```python
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# scale the features
X_scaler = StandardScaler()
X_train = pd.DataFrame(X_scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(X_scaler.fit_transform(X_test), columns=X_test.columns)

y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train[:, None])[:, 0]
y_test = y_scaler.transform(y_test[:, None])[:, 0]

lm = LinearRegression()
lm.fit(X_train, y_train)

predicted = lm.predict(X_test)
print ("Test Accuracy:", round(metrics.r2_score(y_test, predicted) * 100, 2), '%')
print ("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predicted),5))

#Plot the co-efficients
coefs = pd.Series(lm.coef_[0], index=X_train.columns)
plt.subplot(2,1,2)
coefs.plot(kind="bar")
plt.show()
```

Test Accuracy: 79.2 %

Mean Squared Error: 0.20952

![features-target-correlation](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/correlation-plot.png?raw=true)

We have brough down the MSE from 10 to 0.2 by scaling the features. All features seem to affect the co-efficient equally.

PLOT true vs predicted scores and draw the line of fit

```python
plt.scatter(y_test, predicted)
plt.plot([-4, 4], [-4, 4], '--k')
plt.xlabel("True overall score")
plt.ylabel("Predicted overall score")
plt.title("True vs Predicted overall score")
plt.show()
```

![defensive rate value counts](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/true-vs-predicted-scores.png?raw=true)
