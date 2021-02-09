import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from io import StringIO
from IPython.display import Image, display
import pydotplus

sns.set()

test_df = pd.read_csv('titanic-test.csv')
train_df = pd.read_csv('titanic-train.csv')

label_encoder = preprocessing.LabelEncoder()

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('S')

train_predictors = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)

categorical_cols = [cname for cname in train_predictors.columns if train_predictors[cname].nunique() < 10 and train_predictors[cname].dtype == 'object']

numerical_cols = [cname for cname in train_predictors.columns if train_predictors[cname].dtype in ['int64', 'float64']]

cols = categorical_cols = numerical_cols

train_predictors = train_predictors[cols]

dummy_encoded_train_predictors = pd.get_dummies(train_predictors)

y_target = train_df['Survived'].values
x_features_one = dummy_encoded_train_predictors.values

X_train, X_test, Y_train, Y_test = train_test_split(x_features_one, y_target, test_size=0.2, random_state=1)

tree_one = tree.DecisionTreeClassifier()
tree_one = tree_one.fit(X_train, Y_train)
tree_one_accuracy = round(tree_one.score(X_test, Y_test), 4)
print(tree_one_accuracy)

out = StringIO()
tree.export_graphviz(tree_one, out_file=out)

graph = pydotplus.graph_from_dot_data(out.getvalue())
graph.write_png('titanic.png')