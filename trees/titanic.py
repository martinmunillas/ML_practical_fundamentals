import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

sns.set()

test_df = pd.read_csv('titanic-test.csv')
train_df = pd.read_csv('titanic-train.csv')

train_df.Sex.value_counts().plot(kind='bar', color=['b', 'r'])
plt.title('Survivor Distribution')

label_encoder = preprocessing.LabelEncoder()

encoder_sex = label_encoder.fit_transform(train_df['Sex'])

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna('S')

train_predictors = train_df.drop('PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', axis=1)

categorical_cols = 

plt.show()