'''Model'''
from os import pread
from random import Random
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

clf1 = RandomForestClassifier(criterion='gini', max_depth=4, n_jobs=-1)
clf2 = LogisticRegression(C=0.1)
clf3 = DecisionTreeClassifier(criterion='gini', max_depth=4)
clf4 = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')


pipe2 = Pipeline([('sc', StandardScaler()),
                  ('clf_log', clf2)])

mv_clf = VotingClassifier(estimators=[('rt', clf1), ('lr', pipe2), ('dt', clf3), ('kn', clf4)], voting='hard')

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'rt__criterion': ['gini', 'entropy', 'log_loss'], 'rt__max_depth': [2, 6, 10]},
              {'lr__clf_log__penalty': ['l2'], 'lr__clf_log__C': param_range},
              {'dt__criterion': ['gini', 'entropy', 'log_loss'], 'dt__max_depth': [2, 6, 10]},
              {'kn__n_neighbors': [2, 4, 6, 8, 10], 'kn__p': [2, 4, 6, 8, 10]}]
gs_1 = GridSearchCV(estimator=mv_clf,
                    param_grid=param_grid,
                    scoring='accuracy',
                    cv=5,
                    n_jobs=-1)


'''Datasets'''
import numpy as np
import pandas as pd

df = pd.read_csv('train.csv', header=None)
df_test = pd.read_csv('test.csv', header=None)

df = df.iloc[:, [0, 1, 2, 4, 5, 6, 7, 9, 11]]
sex_map = {'male': 0, 'female': 1}
embarked_map = {'S': 0, 'C': 1}
df.iloc[1:, 3] = df.iloc[1:, 3].map(sex_map)
df.iloc[1:, 8] = df.iloc[1:, 8].map(lambda x: embarked_map.get(x, 2))

age_df = df.iloc[:, 2:]
known_age = age_df[age_df.iloc[:, 2].notna()]
unknown_age = age_df[age_df.iloc[:, 2].isna()]

X_age = known_age.iloc[1:, [0, 1, 3, 4, 5, 6]].values
y_age = known_age.iloc[1:, 2].values

rf_age = RandomForestClassifier(random_state=0, n_estimators=100, n_jobs=-1)
rf_age.fit(X_age, y_age)
predAge = rf_age.predict(unknown_age.iloc[:, [0, 1, 3, 4, 5, 6]].values)

df.iloc[1:, 4][df.iloc[1:, 4].isnull()] = predAge


X = df.iloc[1:, 2:].values
y = df.iloc[1:, 1].astype(int).values


df_test = df_test.iloc[:, [0, 1, 3, 4, 5, 6, 8, 10]]
df_test.iloc[1:, 2] = df_test.iloc[1:, 2].map(sex_map)
df_test[10] = df_test[10].map(lambda x: embarked_map.get(x, 2))
df_test.iloc[1:, 6] = df_test.iloc[1:, 5].fillna(df_test.iloc[1:, 5].median())


age_df_test = df_test.iloc[:, 1:]

known_age_test = age_df_test[age_df_test.iloc[:, 2].notna()]
unknown_age_test = age_df_test[age_df_test.iloc[:, 2].isna()]

X_age_test = known_age_test.iloc[1:, [0, 1, 3, 4, 5, 6]].values
y_age_test = known_age_test.iloc[1:, 2].values

rf_age.fit(X_age_test, y_age_test)

y_pred_age = rf_age.predict(unknown_age_test.iloc[:, [0, 1, 3, 4, 5, 6]].values)
df_test.iloc[1:, 3][df_test.iloc[1:, 3].isnull()] = y_pred_age

X_test = df_test.iloc[1:, 1:].values.astype(float)

'''fit and predict'''
clf1.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)
gs_1.fit(X, y)
y_pred = gs_1.predict(X_test)


'''make submission'''
pass_id = df_test.iloc[1:, 0].values.astype(int)
submit = pd.DataFrame(data={'PassengerId': pass_id,
                            'Survived': y_pred})
submit.to_csv('submission_Ensemble_Age.csv', index=None)


'''ハイパーパラメータ適合後の正解率
results = gs_1.cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print(mean_score, params)
'''
