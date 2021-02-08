import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('salaries.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()

regressor.fit(X_train, Y_train)

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='black')
plt.show()


print(regressor.score(X_test, Y_test))