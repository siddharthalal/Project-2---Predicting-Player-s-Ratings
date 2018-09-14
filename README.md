# Predicting-Player's-Ratings

In this project, we are going to predict the overall rating of soccer player based on their attributes such as 'crossing', 'finishing etc.

The dataset that we are going to use is from the European Soccer Database (https://www.kaggle.com/hugomathien/soccer). 

We will be building two different models using Linear and Random Forest regression techniques, and evalaute those models to judge their accuracy and efficiency.

---

Let’s first import some libraries that we are going to need for our analysis:

```python
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from math import sqrt
from matplotlib import pyplot as plt
import seaborn as sns
# allow plots to appear directly in the notebook
%matplotlib inline
```

Load the .sqlite file into a Pandas data frame.

```python
cnxcnx  ==  sqlite3sqlite3.connect('players-database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.head()
```

![dataframe](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/dataframe.png?raw=true)

### Data clean up

```python
# 836 records which have "overall_rating" as null have null values in all the columns. We can safely drop them.
df = df[~df.overall_rating.isnull()]

# replace NANs with series means

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

### Data Pre-processing 

```python
#Dummify the categorical features and Normalize the numeric features

df_category = df[['attacking_work_rate','defensive_work_rate', 'preferred_foot']]
df_numeric = df[np.setdiff1d(df.columns.tolist(), df_category.columns.tolist())]
df_category_dummified = pd.get_dummies(df_category, columns=['attacking_work_rate','defensive_work_rate', 'preferred_foot'])
cols_to_norm = df_numeric.columns.tolist()
df_numeric[cols_to_norm] = df_numeric[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Modified data frame
df_modified = pd.concat([df_category_dummified, df_numeric], axis=1)
```

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

### Linear Regression

```python
#Evaluate the model by splitting into train and test sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
predicted
print ("Test Accuracy:", round(metrics.r2_score(y_test, predicted) * 100, 2), '%')
print ("Mean Squared Error:", metrics.mean_squared_error(y_test, predicted))
```

Test Accuracy: 79.29 %

Mean Squared Error: 0.0027657257983778115

Evaluate the model using 10-fold cross-validation

```python
scores = cross_val_score(LinearRegression(), X, y, cv=10)
scores, scores.mean()
```

(array([0.80033059, 0.7976755 , 0.78623793, 0.78128898, 0.78732932,
        0.77604425, 0.81277596, 0.78589619, 0.78631669, 0.79532529]),
 0.7909220676348424)
 
Accuracy still at 79%.

PLOT true vs predicted scores and draw the line of fit

```python
pltplt..scatterscatter((y_testy_test,,  predictedpredicte )
plt.plot([0, 1], [0, 1], '--k')
plt.xlabel("True overall score")
plt.ylabel("Predicted overall score")
plt.title("True vs Predicted overall score")
plt.show()
```

![defensive rate value counts](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/true-vs-predicted-scores.png?raw=true)

Draw residual plot. If the data points are scattered randomly around the line, then our model is correct and it's not missing the relationship between any two features.

```python
plt.figure(figsize=(9,6))
plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, c='b', s=40, alpha=0.5)
plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, c='g', s=40, alpha=0.5)
plt.hlines(y=0, xmin=0, xmax=1)
plt.ylabel('Residuals')
plt.title('Residual plot including training(blue) and test(green) data')
plt.show()
```

![defensive rate value counts](https://github.com/siddharthalal/Project-2---Predicting-Player-s-Ratings/blob/master/residual%20plot.png?raw=true)
