# -------- module install --------
from random import random
from select import KQ_NOTE_RENAME
from tabnanny import verbose
from traceback import print_tb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math




# -------- datasets --------
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_test['SalePrice'] = np.nan
df = pd.concat([df_train, df_test], ignore_index=True)



# -------- nullの補間 --------
df['Alley'].fillna('Nodata', inplace=True)
df['BsmtQual'].fillna('Nodata', inplace=True)
df['BsmtCond'].fillna('Nodata', inplace=True)
df['BsmtExposure'].fillna('Nodata', inplace=True)
df['BsmtFinType1'].fillna('Nodata', inplace=True)
df['BsmtFinSF1'].fillna(0, inplace=True)
df['BsmtFinType2'].fillna('Nodata', inplace=True)
df['BsmtFinSF2'].fillna(0, inplace=True)
df['BsmtUnfSF'].fillna(0, inplace=True)
df['TotalBsmtSF'].fillna(0, inplace=True)
df['Utilities'].fillna('AllPub', inplace=True)
df['BsmtFullBath'].fillna(0.0, inplace=True)
df['BsmtHalfBath'].fillna(0.0, inplace=True)
df['FireplaceQu'].fillna('Nodata', inplace=True)
df['GarageType'].fillna('Nodata', inplace=True)
df['GarageYrBlt'].fillna(0.0, inplace=True)
df['GarageFinish'].fillna('Nodata', inplace=True)
df['GarageQual'].fillna('Nodata', inplace=True)
df['GarageCond'].fillna('Nodata', inplace=True)
df['GarageCond'].fillna('Nodata', inplace=True)
df.loc[(df['PoolArea']!=0) & (df['PoolQC'].isnull()), 'PoolQC'] = 'Fa'
df['PoolQC'].fillna('Nodata', inplace=True)
df['Fence'].fillna('Nodata', inplace=True)
df['MiscFeature'].fillna('Nodata', inplace=True)

df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)
df['KitchenQual'].fillna(df['KitchenQual'].mode()[0], inplace=True)
df['Electrical'].fillna(df['Electrical'].mode()[0], inplace=True)
df['Functional'].fillna(df['Functional'].mode()[0], inplace=True)
df['SaleType'].fillna(df['SaleType'].mode()[0], inplace=True)
df['Exterior1st'].fillna(df['Exterior1st'].mode()[0], inplace=True)
df['Exterior2nd'].fillna(df['Exterior2nd'].mode()[0], inplace=True)
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace=True)
df['MasVnrArea'].fillna(0.0, inplace=True)

df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.mean()))
df['GarageCars'] = df.groupby('GarageType')['GarageCars'].transform(lambda x: x.fillna(x.mean()))
df['GarageArea'] = df.groupby('GarageType')['GarageArea'].transform(lambda x: x.fillna(x.mean()))

df['MSSubClass'] = df['MSSubClass'].astype(str)



# -------- 特徴量抽出 --------
df['HasGarage'] = np.where(df['GarageQual']=='Nodata', 0, 1)
df['HasPool'] = np.where(df['PoolQC']=='Nodata', 0, 1)
df['HasMiscFeature'] = np.where(df['MiscFeature']=='Nodata', 0, 1)
df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
df['RmSFAbvGrd'] = (df['1stFlrSF'] + df['2ndFlrSF']) / df['TotRmsAbvGrd']
df['TotalFullBath'] = df['FullBath'] + df['BsmtFullBath']
df['TotalHalfBath'] = df['HalfBath'] + df['BsmtHalfBath']
df['TotalBath'] = df['TotalFullBath'] + df['TotalHalfBath']
df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
df['YrPsdBltnRmd'] = df['YearBuilt'] + df['YearRemodAdd']



# -------- Label Encoding --------
# Target Encoding
def target_encoding(column):
    col_sorted = df.groupby(column)['SalePrice'].mean().sort_values()
    col_index = col_sorted.index.values
    col_dict = dict(zip(col_index, range(len(col_index))))
    df[column] = df[column].map(lambda x: col_dict[x])



df_object = df.loc[:, df.dtypes=='object'].columns.values
for i in df_object:
    target_encoding(i)

df['YearBuilt'] = 2023 - df['YearBuilt']
df['YearRemodAdd'] = 2023 - df['YearRemodAdd']
df['GarageYrBlt'] = 2023 - df['GarageYrBlt']
df['YrSold'] = 2023 - df['YrSold']



# -------- 特徴量選択 --------
def remove_high_correlation_columns(df, corr_threshold=0.85):
    corrmat = df.corr()
    corr_high = np.where((corrmat>=corr_threshold) & (corrmat != 1.0))
    high_corr_indices = list(zip(corr_high[0], corr_high[1]))
    high_corr_indices = [(i, j) for i, j in high_corr_indices if i < j]

    
    drop_index = []
    for tap in high_corr_indices:
        drop_index.append(tap[0])
    drop_index = list(set(drop_index))
    df.drop(df.columns[drop_index], axis=1, inplace=True)
    
    return df

df = remove_high_correlation_columns(df, 0.95) # チューニング可




# -------- 評価関数 --------

# RMSE から RMSLEに変換
# plt.hist(df['SalePrice'], bins=100, stacked=True)
# plt.xlabel('RMSE')
# plt.show()
# print(df['SalePrice'].skew(), df['SalePrice'].kurt())
df['SalePrice'] = np.log(df['SalePrice'] + 1)

# plt.hist(df['SalePrice'], bins=100, stacked=True)
# plt.xlabel('RMSLE')
# plt.show()
# print(df['SalePrice'].skew(), df['SalePrice'].kurt())



# -------- 適用データ作成 --------
df_train_X = df.loc[df['SalePrice'].notna(), df.drop('SalePrice', axis=1).columns]
df_train_y = df.loc[df['SalePrice'].notna(), 'SalePrice']

df_test_X = df.loc[df['SalePrice'].isna(), df.drop('SalePrice', axis=1).columns]
df_test_y = df.loc[df['SalePrice'].isna(), 'SalePrice']



from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
sc2 = StandardScaler()

df_train_X_std = sc1.fit_transform(df_train_X)
df_test_X_std = sc2.fit_transform(df_test_X)

model_en = ElasticNet(alpha=0.001339195717010918, l1_ratio = 0.6880179381963745)
model_en.fit(df_train_X, df_train_y)
en_pred = model_en.predict(df_test_X)

en_pred = np.exp(en_pred) - 1


model_en_std = ElasticNet(alpha=0.001339195717010918, l1_ratio = 0.6880179381963745)
model_en_std.fit(df_train_X_std, df_train_y)
en_pred_std = model_en_std.predict(df_test_X_std)

en_pred_std = np.exp(en_pred_std) - 1

print(en_pred)
print(en_pred_std)

a = []

