# FILTER METHOD-BASED FEATURE SELECTION
# This includes correlation and mutual Information
# https://towardsdatascience.com/feature-selection-in-python-using-filter-method-7ae5cbc4ee05
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

# Reading the data
cardata = pd.read_csv('mpg.csv')
print(cardata.dropna())
print(cardata.shape)
print(cardata.columns)
cardata.head(2)
cardata.info()  # Horsepower and name as object

# We see that horsepower is no more a categorical variable and car name is the only categorical variable
# We check the presence of categorical features
cardata.describe(include='O')
# Updating the horsepower feature to integer and filling all nulls with 0
cardata['horsepower'] = pd.to_numeric(cardata['horsepower'], errors='coerce').fillna(0).astype(int)
# We now check again for the presence of categorical variables
cardata.describe(include='O')
# We see that horsepower is no more a categorical variable and Car name is the only categorical variable.

# LABEL ENCODING
# Creating a labelEncoder for Car name to encode Car names with a value between 0 and n_classes-1. In our case n_classes for Car name is 305
labelencoder = LabelEncoder()
X_en = cardata.iloc[:,8].values
X_en = labelencoder.fit_transform(X_en)
# Creating the input features X and target variable y
X = cardata.iloc[:,1:7]
X['name'] = X_en
Y = cardata.iloc[:,0].values
# Create a data set with all the input features after converting them to numeric including target variable
fulldata = X.copy()
fulldata['mpg'] = Y
fulldata.head(2)

# STEP 1 Identify input features having high correlation with target variable.
importances = fulldata.drop("mpg", axis=1).apply(lambda x: x.corr(fulldata.mpg))
indices = np.argsort(importances)
print(importances[indices])
# --- Plotting this data for visualization
names = ['cylinders','displacement','horsepower','weight','acceleration','model_year', 'car']
plt.title('Miles per Gallon')
plt.barh(range(len(indices)), importances[indices], color = 'g', align='center')
plt.yticks(range(len(indices)), [names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
# We set the threshold to the absolute value of 0.4. We keep input features only if the correlation of the input feature with the target variable is greater than 0.4
for i in range (0, len(indices)):
    if np.abs(importances[i]) > 0.4:
        print(names[i])
# We now have reduced the input features from 7 to 6. Car name was dropped as it was not having a high correlation with mpg(miles per gallon)
X = cardata[ ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']]

# STEP 2 Identify input features that have a low correlation with other independent variables.
# --- We will keep input features that are not highly correlated with other input features
for i in range(0,len(X.columns)):
    for j  in range(0, len(X.columns)):
        if i!=j:
            corr_1 = np.abs(X[X.columns[i]].corr(X[X.columns[j]]))
            if corr_1 < 0.3:
                print(X.columns[i], "is not correlated with", X.columns[j])
            elif corr_1 > 0.75:
                print(X.columns[i], "is highly correlated with", X.columns[j])
# displacement, horsepower, cylinder, and weight are highly correlated. We will keep only keep one of them.
X = cardata[['cylinders', 'acceleration','weight']]

# STEP 3 Find the information gain or mutual information of the independent variable with respect to a target variable
mi = mutual_info_regression(X, Y)
# Plotting the mutual information
mi = pd.Series(mi)
mi.index = X.columns
mi_srt = mi.sort_values(ascending=False)
print(mi_srt)
