import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


df['Age'].fillna(df['Age'].mean(), inplace=True)
df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)

df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
df_test.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

category_list = ['Name', 'Cabin', 'Ticket','Embarked']

df.drop(category_list, axis=1, inplace=True)
df_test.drop(category_list, axis=1, inplace=True)


X = df[['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare']].values
Y = df['Survived'].values

model = DecisionTreeClassifier(max_depth=3)
result = model.fit(X, Y)

X_test = df_test.iloc[:, 1:].values

predict = model.predict(X_test)

submit_csv = pd.concat([df_test['PassengerId'], pd.Series(predict)], axis=1)
submit_csv.columns = ['PassengerId', 'Survived']
submit_csv.to_csv('submition.csv', index=False)