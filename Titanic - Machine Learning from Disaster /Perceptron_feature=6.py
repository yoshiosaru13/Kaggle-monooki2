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

X = df.iloc[1:, 2:7].values
y = df.iloc[1:, 1].astype(int).values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X)

print(X_std)

