__author__ = 'ZkHaider'

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import time
import pandas as pd

X = pd.read_csv('csv/titanic_train.csv')
y = X.pop('Survived')

# Describe the data
print X.describe()

# Input missing Age values with the mean value
X['Age'].fillna(X.Age.mean(), inplace=True)

# Confirm the code is correct
#print X.describe()

# Get just the numeric values by selecting only the variables that are not 'object' datatypes
numeric_variables = list(X.dtypes[X.dtypes != 'object'].index)
#print X[numeric_variables].head()

# Build RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)

# Only use numerical variables for now since categorical data needs to be fixed
model.fit(X[numeric_variables], y)

# For regression oob_score attribute will give us the R^2 value
y_oob = model.oob_prediction_

# Print accuracy score without optimization
#print "C-Stat: ", roc_auc_score(y, y_oob)

# Drop variables that would seem useless to this dataset, this is data that is similar or identical
X.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return 'None'

X['Cabin'] = X.Cabin.apply(clean_cabin)

categorical_variables = ['Sex', 'Cabin', 'Embarked']

for variable in categorical_variables:
    # Fill missing data with the word "Missing"
    X[variable].fillna('Missing', inplace=True)
    # Create array of dummies
    dummies = pd.get_dummies(X[variable], prefix=variable)
    # Update X to include dummies and drop the main variable
    X = pd.concat([X, dummies], axis=1)
    X.drop([variable], axis=1, inplace=True)

# Rebuild model
model = RandomForestRegressor(100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X, y)

# Print the new C-Stat accuracy score
new_y_oob = model.oob_prediction_
#print "C-Stat: ", roc_auc_score(y, new_y_oob)

# Save the plot
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort()
ax = feature_importances.plot(kind='barh', figsize=(7, 6))
fig = ax.get_figure()
fig.savefig('feature_importances.png')

def graph_feature_importances(model, feature_names, autoscale=True, headroom=0.05, width=10, summarized_columns=None):
    if autoscale:
        x_scale = model.feature_importances_.max() + headroom
    else:
        x_scale = 1

    feature_dict = dict(zip(feature_names, model.feature_importances_))
    if summarized_columns:
        for col_name in summarized_columns:
            # Summarize some dummy columns, sum all features that contain col_name, store in temp sum_value
            sum_value = sum(x for i, x in feature_dict.iteritems() if col_name in i)
            # Remove all keys that are part of col_name
            keys_to_remove = [i for i in feature_dict.keys() if col_name in i]
            for i in keys_to_remove:
                feature_dict.pop(i)
            # Read the summarized field
            feature_dict[col_name] = sum_value

    results = pd.Series(feature_dict.values(), index=feature_dict.keys())
    results.sort(axis=1)
    new_ax = results.plot('barh', figsize=(width, len(results) / 4), xlim=(0, x_scale))
    new_fig = new_ax.get_figure()
    new_fig.savefig('condensed_feature_importances.png')

graph_feature_importances(model, X.columns, summarized_columns=categorical_variables)

# Optimize the model - These are optionals code blocks used to check which modifiers work the best

## start with optimization of n_estimators
# n_estimator_results = []
# n_estimator_options = [30, 50, 100, 200, 500, 1000, 2000]
#
# for trees in n_estimator_options:
#     model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
#     model.fit(X, y)
#     print trees, 'trees'
#     roc = roc_auc_score(y, model.oob_prediction_)
#     print 'C-State: ', roc
#     n_estimator_results.append(roc)
#     print ''
#
# estimator_plot = pd.DataFrame(n_estimator_results, n_estimator_options).plot(kind='barh', xlim=(0.85, .87))
# estimator_plot.get_figure().savefig('opt_estimator.png')
#
# ## optimize max_features
# max_features_results = []
# max_features_options = ['auto', None, 'sqrt', 'log2', 0.9, 0.2]
#
# for max_features in max_features_options:
#     model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
#     model.fit(X, y)
#     print max_features, 'max_features'
#     roc = roc_auc_score(y, model.oob_prediction_)
#     print 'C-State: ', roc
#     max_features_results.append(roc)
#     print ''
#
# feature_plot = pd.Series(max_features_results, max_features_options).plot(kind='barh', xlim=(0.86, .87))
# feature_plot.get_figure().savefig('opt_features.png')
#
# ## optimize min leafs
# min_leafs_results = []
# min_leafs_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
# for min_sample in min_leafs_options:
#     model = RandomForestRegressor(n_estimators=1000,
#                                   oob_score=True,
#                                   n_jobs=-1,
#                                   random_state=42,
#                                   max_features='auto',
#                                   min_samples_leaf=min_sample)
#     model.fit(X, y)
#     print min_sample, 'min samples'
#     roc = roc_auc_score(y, model.oob_prediction_)
#     print 'C-Stat: ', roc
#     min_leafs_results.append(roc)
#     print ''
#
# min_leaf_plot = pd.Series(min_leafs_results, min_leafs_options).plot(kind='barh', xlim=(0.85, 0.88))
# min_leaf_plot.get_figure().savefig('opt_min_leaf.png')

## Based on our optimization we can see that having 100 estimators, auto on max_features and min_leaf sample of 5

model = RandomForestRegressor(n_estimators=100,
                              oob_score=True,
                              n_jobs=-1,
                              random_state=42,
                              max_features='auto',
                              min_samples_leaf=5)

model.fit(X, y)
roc = roc_auc_score(y, model.oob_prediction_)
print 'C-Stat: ', roc





