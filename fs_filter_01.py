# FILTER METHOD-BASED FEATURES SELECTION
# --- https://medium.com/mlearning-ai/feature-selection-using-filter-method-python-implementation-from-scratch-375d86389003
# --- No Constant Features
# --- No Quasi-Constant Features
# --- No Duplicate Features

# Import Library
import pandas as pd
import numpy as np
from pandas import Index
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

# Read the dataset
data = pd.read_csv(r"Standard Customer Data.csv", nrows=40000)
print("Number of Data :",data.shape)    # Include name of features
train_x, test_x, train_y, test_y= train_test_split(
    data.drop("TARGET",axis=1),
    data.TARGET,
    test_size=0.2,
    random_state=41)
# Here, X = data.drop("TARGET",axis=1), Y = data.Target

# STEP 1 - REMOVING CONSTANT FEATURES
print("REMOVING CONSTANT FEATURES -----------------------------------------")
constant_filter = VarianceThreshold(threshold=0)
# --- Fit and transforming on training data
data_constant = constant_filter.fit_transform(train_x)
print(data_constant.shape)
# --- Extracting all constant columns using get support function of our filter
constant_columns = [column for column in train_x.columns
                    if column not in train_x.columns[constant_filter.get_support()]]
# --- Number of constant columns
print("Number of Constant Columns :",len(constant_columns))
# --- Constant columns names:
for column in constant_columns:
    print("Constant columns name :", column)
# --- Removing the above-identified constant columns from our database
data_cons = data.drop(constant_columns, axis=1)
print(data_cons.shape)      # with feature name of rows

# STEP 2 - REMOVING QUASI-CONSTANT FEATURES
print("REMOVING QUASI-CONSTANT FEATURES -----------------------------------")
qcons_filter = VarianceThreshold(threshold=0.01)
# --- Fit and transforming on train data
data_qcons = qcons_filter.fit_transform(train_x)
print(data_qcons.shape)
# --- Extracting all Quasi constant columns using get support function of our filter
qcons_columns = [column for column in train_x.columns
                    if column not in train_x.columns[qcons_filter.get_support()]]
# --- Number of Quasi constant columns
print("Number of Quasi-constant Columns :", len(qcons_columns))
# --- Quasi Constant columns names:
for column in qcons_columns:
    print("Quasi-constant Columns name :", column)
# --- Removing above-identified quasi constant columns from our database
data_qcons = data.drop(qcons_columns, axis=1)
print(data_qcons.shape)

# STEP 3 - REMOVE DUPLICATE COLUMNS
print("REMOVING DUPLICATE FEATURES ----------------------------------------")
# --- Transposing our “quasi-constant” modified dataset.
data_qcons_t = data_qcons.T
print(data_qcons_t.shape)
# --- Now, that our columns have taken the place of the row, we can find the duplicacy in columns:
print("Number of columns to be removed :",data_qcons_t.duplicated().sum())
# --- We have ... more columns to be removed that are duplicated.
# --- Dropping Duplicated method using drop_duplicates()
data_qcons_dup = data_qcons_t.drop_duplicates(keep='first').T
print("Number of training set :",data_qcons_dup.shape)
# We got a better-refined training set with 245 columns now.
