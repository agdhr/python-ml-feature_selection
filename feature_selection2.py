# CORRELATION-BASED FEATURE SELECTION
# - https://johfischer.com/2021/08/06/correlation-based-feature-selection-in-python-from-scratch/
# Objectives : reduce model complexity, enhance learning efficiency, increase predictive power (by reducing noise)
# CFS is a filter approach and therefore independent of the final classification model.
# CFS evaluates feature subsets only based on data intrinsic properties - named correlations
# Goal: to find a feature subset with low feature-feature correlation, to avoid redundancy, and high feature-class correlation to maintain or increase predictive power
# Merit = k * rcf / sqrt(k + k(k-1)*rff)
# rff: average feature - feature correlation
# rrf: average feature - class correlation
# k: number of features of that subset

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import svm
import time

# load the dataset
df = pd.read_csv('madelon.csv')
df.head(3)
# name of the label (can be seen in the dataframe)
label = 'Class'
# list with feature names (V1, V2, V3, ...)
features = df.columns.tolist()
features.remove(label)
# change class labeling to 0 and 1
df[label] = np.where( df[label] > 1, 1, 0)
from scipy.stats import pointbiserialr
from math import sqrt

def getMerit(subset, label):
    k = len(subset)
    # average feature-class correlation
    rcf_all = []
    for feature in subset:
        coeff = pointbiserialr( df[label], df[feature] )
        rcf_all.append( abs( coeff.correlation ) )
    rcf = np.mean( rcf_all )
    # average feature-feature correlation
    corr = df[subset].corr()
    corr.values[np.tril_indices_from(corr.values)] = np.nan
    corr = abs(corr)
    rff = corr.unstack().mean()
    return (k * rcf) / sqrt(k + k * (k-1) * rff)
subset = ['V1', 'V2', 'V3', 'V4']
d = getMerit(subset, label)
corr = df[subset].corr()
corr.values[np.tril_indices_from(corr.values)] = np.nan
print(corr)

best_value = -1
best_feature = ''
for feature in features:
    coeff = pointbiserialr( df[label], df[feature] )
    abs_coeff = abs( coeff.correlation )
    if abs_coeff > best_value:
        best_value = abs_coeff
        best_feature = feature

print("Feature %s with merit %.4f"%(best_feature, best_value))


class PriorityQueue:
    def __init__(self):
        self.queue = []
    def isEmpty(self):
        return len(self.queue) == 0
    def push(self, item, priority):
        """
        item already in priority queue with smaller priority:
        -> update its priority
        item already in priority queue with higher priority:
        -> do nothing
        if item not in priority queue:
        -> push it
        """
        for index, (i, p) in enumerate(self.queue):
            if (set(i) == set(item)):
                if (p >= priority):
                    break
                del self.queue[index]
                self.queue.append((item, priority))
                break
        else:
            self.queue.append((item, priority))
    def pop(self):
        # return item with highest priority and remove it from queue
        max_idx = 0
        for index, (i, p) in enumerate(self.queue):
            if (self.queue[max_idx][1] < p):
                max_idx = index
        (item, priority) = self.queue[max_idx]
        del self.queue[max_idx]
        return (item, priority)
# initialize queue
queue = PriorityQueue()
# push first tuple (subset, merit)
queue.push([best_feature], best_value)
# list for visited nodes
visited = []
# counter for backtracks
n_backtrack = 0
# limit of backtracks
max_backtrack = 5

# repeat until queue is empty
# or the maximum number of backtracks is reached
while not queue.isEmpty():
    # get element of queue with highest merit
    subset, priority = queue.pop()

    # check whether the priority of this subset
    # is higher than the current best subset
    if (priority < best_value):
        n_backtrack += 1
    else:
        best_value = priority
        best_subset = subset

    # goal condition
    if (n_backtrack == max_backtrack):
        break

    # iterate through all features and look of one can
    # increase the merit
    for feature in features:
        temp_subset = subset + [feature]

        # check if this subset has already been evaluated
        for node in visited:
            if (set(node) == set(temp_subset)):
                break
        # if not, ...
        else:
            # ... mark it as visited
            visited.append(temp_subset)
            # ... compute merit
            merit = getMerit(temp_subset, label)
            # and push it to the queue
            queue.push(temp_subset, merit)
    print(subset)
    print("Number of Feature = %s with merit %.10f" % (len(subset), merit))
