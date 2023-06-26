import numpy as np
import pandas as pd
df = pd.read_csv('train.csv', header=None)

#preprocessing#
df = df.dropna(subset=5)
X = df.iloc[1:, [2, 5]].values.astype(float).round().astype(int)
y = df.iloc[1:, 1].values.astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)

from sklearn.metrics import accuracy_score
#print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


df2 = pd.read_csv('test.csv', header=None)
df2 = df2.iloc[0:, [0, 1, 4]]

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
df2_imp = imp.fit_transform(df2.iloc[1:]).astype(int)

X_test_train = df2_imp[0:, [1, 2]]
X_test_train_std = sc.fit_transform(X_test_train)
y_test_pred = lr.predict(X_test_train_std)

submit = pd.DataFrame(data={'PassengerId': df2_imp[:, 0],
                            'Survived': y_test_pred})
submit.to_csv('submition2.csv', index=False)