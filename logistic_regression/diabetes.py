import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

diabetes = pd.read_csv('diabetes.csv')
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
x = diabetes[feature_cols]
y = diabetes.Outcome

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(Y_test, y_pred)

class_names = [0,1]
fix, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues_r', fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual')
plt.xlabel('Prediction')

print('Accuracy:', metrics.accuracy_score(Y_test, y_pred))

plt.show()