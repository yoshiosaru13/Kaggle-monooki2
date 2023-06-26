# -------- module install --------
from random import random
from select import KQ_NOTE_RENAME
from tabnanny import verbose
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


from sklearn.linear_model import ElasticNet,Lasso,BayesianRidge,LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_splits=4, shuffle=True, random_state=8)
    rmse= np.sqrt(-cross_val_score(model, df_train_X, df_train_y, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# LASSO回帰
lasso = make_pipeline(RobustScaler(),Lasso(alpha=0.0005,random_state=1))
# Elastic Net
ENet = make_pipeline(RobustScaler(),ElasticNet(alpha=0.0005,l1_ratio=.9,random_state=3))
# Kernel Ridge
KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6,kernel='polynomial',degree=2,coef0=1))
# GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000,
                                   learning_rate=0.05,
                                   max_depth=4,
                                   max_features='sqrt',
                                   min_samples_leaf=15,
                                   min_samples_split=10,
                                   loss='huber'
                                   ,random_state=5)
# XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
# LightGBM
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


score = rmsle_cv(lasso)
print('Lasso score:{:.4f}({:.4f})'.format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print('Kernel Ridge score:{:.4f}({:.4f})'.format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print('Gradient Boosting score:{:.4f}({:.4f})'.format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print('Xgboost score:{:.4f}({:.4f})'.format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print('LGBM score:{:.4f}({:.4f})'.format(score.mean(), score.std()))

#平均化モデルのスタッキング
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
  
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
#平均化モデルのスタッキングのスコア
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
                                                 meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)
print('Stacking Averaged base models score:{:.4f}({:.4f})'.format(score.mean(), score.std()))
