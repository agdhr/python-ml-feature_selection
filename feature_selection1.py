# FEATURE SELECTION METHOD
# Feature selection is the process of reducing the number of input variables (features) when developing a predictive model.
# Feature selection methods are intended to reduce the number of input variables to those that are believed to be most useful to a model in order to predict the target variable.
# - https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
from matplotlib import pyplot
import pandas as pd
from pandas import DataFrame, read_csv
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

filename = 'breast-cancer.csv'
# Load the dataset
def load_dataset(filename):
    # - load the dataset as a pandas DataFrame
    data = read_csv(filename, header= None)
    # - retrieve numpy array
    dataset = data.values
    # - Split into input (X) and output (y) variables
    X = dataset[:,:-1]
    y = dataset[:,-1]
    # - format all fields as string
    X = X.astype(str)
    return X, y
# To split the data into training and test sets
X, y = load_dataset(filename)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)

# Prepare input data
def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc

# Prepare target/output data
def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc

# Run to prepare input data
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
# Run to prepare target data
y_train_enc, y_test_enc = prepare_targets(y_train, y_test)

# Feature selection
def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs
X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
# Score for the features
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()
# NOTE: Perhaps features 2, 3, 5, 6, 7, and 8 are most relevant.

# Fit the model
model = LogisticRegression(solver='lbfgs')
# model.fit(X_train_enc, y_train_enc)
model.fit(X_train_fs, y_train_enc)
# Evaluate the model
yhat = model.predict(X_test_enc)
# Evaluate predictions
accuracy = accuracy_score(y_test_enc, yhat)
print('Accuracy: %2f' % (accuracy*100))