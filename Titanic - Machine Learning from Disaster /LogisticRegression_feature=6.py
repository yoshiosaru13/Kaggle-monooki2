import numpy as np
import pandas as pd

df = pd.read_csv('train.csv', header=None)
df_test = pd.read_csv('test.csv', header=None)

df = df.iloc[:, [0, 1, 2, 4, 5, 6, 7, 9]]
sex_map = {'Sex': 'Sex', 'male': 0, 'female': 1}
df.iloc[1:, 3] = df.iloc[1:, 3].map(sex_map)

from sklearn.impute import SimpleImputer
imp = SimpleImputer()
df.iloc[1:, ] = imp.fit_transform(df.iloc[1:, ])

X = df.iloc[1:, 2:].values
y = df.iloc[1:, 1].astype(int).values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0)
lr.fit(X_std, y)

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_score, test_score = learning_curve(estimator=lr,
                                                      X=X_std,
                                                      y=y,
                                                      train_sizes=np.linspace(0.1, 1, 10),
                                                      cv=10,
                                                      n_jobs=-1)

train_mean = np.mean(train_score, axis=1)
train_std = np.std(train_score, axis=1)
test_mean = np.mean(test_score, axis=1)
test_std = np.mean(test_score, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.plot(train_sizes, test_mean, color='r', marker='x', markersize=5, label='test accuracy')
plt.grid()
plt.legend(loc='upper right')
plt.show()

df_test = df_test.iloc[:, [0, 1, 3, 4, 5, 6, 8]]
df_test[3] = df_test[3].map(sex_map)
X_test = df_test.iloc[1:, 1:].values.astype(float)
X_test = imp.fit_transform(X_test)
X_test_std = sc.fit_transform(X_test)

y_pred = lr.predict(X_test_std)
pass_id = df_test.iloc[1:, 0].values.astype(int)

submit = pd.DataFrame(data={'PassengerId': pass_id,
                            'Survived': y_pred})
submit.to_csv('submission_LogisticRegression_f=6.csv', index=None)