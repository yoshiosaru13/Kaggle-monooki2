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




# -------- モデル ---------
from sklearn.model_selection import KFold, cross_validate
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet


# -------- LightGBM --------


#チューニング用のモデルを作成
def objective_lgb(trial):
    params_lgb = {
        'objective':'regression',
        'metric':'r2',
        'verbosity':-1,
        'num_leaves':trial.suggest_int('num_leaves', 10, 1000),
        'max_depth':trial.suggest_int('max_depth', 8, 24),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
        'reg_alpha':trial.suggest_loguniform('reg_alpha', 1e-5, 1.0),
        'reg_lambda':trial.suggest_loguniform('reg_lambda', 1e-5, 1.0),
        'feature_fraction':trial.suggest_uniform('feature_fraction', 0.3, 1.0),
        'bagging_fraction':trial.suggest_uniform('bagging_fraction', 0.3, 1.0),
        'bagging_freq':trial.suggest_int('bagging_freq', 1, 8),
        'min_child_samples':trial.suggest_int('min_child_samples', 5, 80),
        'random_state':8
        }
#見込みのないtrialの切り捨て
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'r2')
#モデルにチューニングしたパラメータを渡す
    model_lgb = lgb.LGBMRegressor(**params_lgb, n_estimators=500, verbose=-1)
#KFold分割し、評価を行う
    kf_lgb = KFold(n_splits=4, shuffle=True, random_state=8)
    scores_lgb = cross_validate(model_lgb, X=df_train_X, y=df_train_y, scoring='r2', cv=kf_lgb)
    return scores_lgb['test_score'].mean()

#Optunaによるチューニング

study = optuna.create_study(direction='maximize')
study.optimize(objective_lgb, timeout=60)
#ベストなパラメーターの表示
best_params = study.best_params
best_params_lgb = best_params


sub_preds_lgb = np.zeros(df_test.shape[0])

kfolds = KFold(n_splits=4, shuffle=True, random_state=1)
model_lgb = lgb.LGBMRegressor(**best_params_lgb)
model_lgb.fit(df_train_X, df_train_y)

scores_lgb = cross_validate(model_lgb, X=df_train_X, y=df_train_y, scoring='neg_mean_squared_error', cv=kfolds)
rmse_lgb = np.sqrt(-scores_lgb['test_score'].mean())


# -------- XGBoost --------

import xgboost as xgb

def objective_xgb(trial):

    #評価するハイパーパラメータの値を規定
    params_xgb ={
        'max_depth':trial.suggest_int("max_depth",1,10),
        'min_child_weight':trial.suggest_int('min_child_weight',1,5),
        'gamma':trial.suggest_uniform('gamma',0,1),
        'subsample':trial.suggest_uniform('subsample',0,1),
        'colsample_bytree':trial.suggest_uniform('colsample_bytree',0,1),
        'reg_alpha':trial.suggest_loguniform('reg_alpha',1e-5,100),
        'reg_lambda':trial.suggest_loguniform('reg_lambda',1e-5,100),        
        'learning_rate':trial.suggest_uniform('learning_rate',0,1)}

    model_xgb = xgb.XGBRegressor(n_estimators=500, n_jobs=-1, random_state=1, **params_xgb)
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'r2')

    #交差検証
    kf_xgb = KFold(n_splits=4, shuffle=True, random_state=8)
    scores_xgb = cross_validate(model_xgb, X=df_train_X, y=df_train_y, scoring='r2', cv=kf_xgb)
    return scores_xgb['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective_xgb, timeout=60)
#ベストなパラメーターの表示
best_params = study.best_params
best_params_xgb = best_params

sub_preds_xgb = np.zeros(df_test_y.shape[0])

kfolds = KFold(n_splits=4, shuffle=True, random_state=1)

model_xgb = xgb.XGBRegressor(**best_params_xgb)
model_xgb.fit(df_train_X, df_train_y)

scores_xgb = cross_validate(model_xgb, X=df_train_X, y=df_train_y, scoring='neg_mean_squared_error', cv=kfolds)
rmse_xgb = np.sqrt(-scores_xgb['test_score'].mean())



# -------- GradientBoostingRegressor ---------
from sklearn.ensemble import GradientBoostingRegressor



def objective_gbr(trial):

    #評価するハイパーパラメータの値を規定
    params_gbr ={
        'n_estimators':trial.suggest_int('n_estimators',2,100),
        'learning_rate':trial.suggest_uniform('learning_rate',0,1),
        'max_depth':trial.suggest_int('max_depth', 8, 24),
        'min_samples_split':trial.suggest_int('min_samples_split',3,7),
        'min_samples_leaf':trial.suggest_int('min_samples_leaf',2,100),
        'max_features':trial.suggest_int('max_features',2,100)}

    model_gbr = GradientBoostingRegressor(random_state=1, **params_gbr)
    

    #交差検証
    kf_gbr = KFold(n_splits=4, shuffle=True, random_state=8)
    scores_gbr = cross_validate(model_gbr, X=df_train_X, y=df_train_y, scoring='r2', cv=kf_gbr)
    return scores_gbr['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective_gbr, timeout=60)
best_params = study.best_params
best_params_gbr = best_params

sub_preds_gbr = np.zeros(df_test_y.shape[0])


kfolds = KFold(n_splits=4, shuffle=True, random_state=1)
    
model_gbr = GradientBoostingRegressor(**best_params_gbr)
model_gbr.fit(df_train_X, df_train_y)

scores_gbr = cross_validate(model_gbr, X=df_train_X, y=df_train_y, scoring='neg_mean_squared_error', cv=kfolds)
rmse_gbr = np.sqrt(-scores_gbr['test_score'].mean())

# --------- Kernel Ridge ---------

def objective_knl(trial):

    #評価するハイパーパラメータの値を規定
    params_knl ={
        'alpha':trial.suggest_loguniform('alpha',1e-5, 100),
        'gamma':trial.suggest_loguniform('gamma',1e-5, 100),
        'degree':trial.suggest_int('degree', 1, 20),
        'coef0':trial.suggest_int('coef0',0,10),
        'kernel':trial.suggest_categorical('kernel',['linear','rbf', 'poly', 'sigmoid'])}
    
    model_knl = KernelRidge(**params_knl)
    

    #交差検証
    kf_knl = KFold(n_splits=4, shuffle=True, random_state=8)
    scores_knl = cross_validate(model_knl, X=df_train_X, y=df_train_y, scoring='r2', cv=kf_knl)
    return scores_knl['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective_knl, timeout=60)
best_params = study.best_params
best_params_knl = best_params
sub_preds_knl = np.zeros(df_test_y.shape[0])


kfolds = KFold(n_splits=4, shuffle=True, random_state=1)


#損失関数
model_knl = KernelRidge(**best_params_knl)
model_knl.fit(df_train_X, df_train_y)

scores_knl = cross_validate(model_knl, X=df_train_X, y=df_train_y, scoring='neg_mean_squared_error', cv=kfolds)
rmse_knl = np.sqrt(-scores_knl['test_score'].mean())


# --------- Elastic Net ---------
def objective_en(trial):

    #評価するハイパーパラメータの値を規定
    params_en ={
        'alpha': trial.suggest_float('alpha', 0.0001, 1, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0, 1)}
    
    model_en = ElasticNet(**params_en)
    

    #交差検証
    kf_en = KFold(n_splits=4, shuffle=True, random_state=8)
    scores_en = cross_validate(model_en, X=df_train_X, y=df_train_y, scoring='neg_mean_squared_error', cv=kf_en)
    return scores_en['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective_en, timeout=60)
best_params = study.best_params
best_params_en = best_params

sub_preds_en = np.zeros(df_test_y.shape[0])


kfolds = KFold(n_splits=4, shuffle=True, random_state=1)

model_en = ElasticNet(**best_params_en)
model_en.fit(df_train_X, df_train_y)

scores_en = cross_validate(model_en, X=df_train_X, y=df_train_y, scoring='neg_mean_squared_error', cv=kfolds)
rmse_en = np.sqrt(-scores_en['test_score'].mean())


print('rmse_lgb', rmse_lgb)
print('rmse_xgb', rmse_xgb)
print('rmse_gbr', rmse_gbr)
print('rmse_knl', rmse_knl)
print('rmse_en', rmse_en)

model_lgb = lgb.LGBMRegressor(**best_params_lgb)
model_lgb.fit(df_train_X, df_train_y)
df_train_pred_lgb = model_lgb.predict(df_train_X,)
df_train_pred_lgb = np.exp(df_train_pred_lgb) - 1
df_pred_lgb = model_lgb.predict(df_test_X)
df_pred_lgb = np.exp(df_pred_lgb) - 1

model_xgb = xgb.XGBRegressor(**best_params_xgb)
model_xgb.fit(df_train_X, df_train_y)
df_train_pred_xgb = model_xgb.predict(df_train_X)
df_train_pred_xgb = np.exp(df_train_pred_xgb) - 1
df_pred_xgb = model_xgb.predict(df_test_X)
df_pred_xgb = np.exp(df_pred_xgb) - 1

model_gbr = GradientBoostingRegressor(**best_params_gbr)
model_gbr.fit(df_train_X, df_train_y)
df_train_pred_gbr = model_gbr.predict(df_train_X)
df_train_pred_gbr = np.exp(df_train_pred_gbr) - 1
df_pred_gbr = model_gbr.predict(df_test_X)
df_pred_gbr = np.exp(df_pred_gbr) - 1

model_knl = KernelRidge(**best_params_knl)
model_knl.fit(df_train_X, df_train_y)
df_train_pred_knl = model_knl.predict(df_train_X)
df_train_pred_knl = np.exp(df_train_pred_knl) - 1
df_pred_knl = model_knl.predict(df_test_X)
df_pred_knl = np.exp(df_pred_knl) - 1

model_en = ElasticNet(**best_params_en)
model_en.fit(df_train_X, df_train_y)
df_train_pred_en = model_en.predict(df_train_X)
df_train_pred_en = np.exp(df_train_pred_en) - 1
df_pred_en = model_en.predict(df_test_X)
df_pred_en = np.exp(df_pred_en) - 1

df_train_base = np.column_stack((df_train_pred_lgb, df_train_pred_xgb, df_train_pred_gbr, df_train_pred_knl, df_train_pred_en))
df_test_base = np.column_stack((df_pred_lgb, df_pred_xgb, df_pred_gbr, df_pred_knl, df_pred_en))



def objective(trial):

    #評価するハイパーパラメータの値を規定
    params ={
        'max_depth':trial.suggest_int("max_depth",1,10),
        'min_child_weight':trial.suggest_int('min_child_weight',1,5),
        'gamma':trial.suggest_uniform('gamma',0,1),
        'subsample':trial.suggest_uniform('subsample',0,1),
        'colsample_bytree':trial.suggest_uniform('colsample_bytree',0,1),
        'reg_alpha':trial.suggest_loguniform('reg_alpha',1e-5,100),
        'reg_lambda':trial.suggest_loguniform('reg_lambda',1e-5,100),        
        'learning_rate':trial.suggest_uniform('learning_rate',0,1)}

    model = xgb.XGBRegressor(n_estimators=500, n_jobs=-1, random_state=1, **params)
    
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'r2')

    #交差検証
    kf = KFold(n_splits=4, shuffle=True, random_state=8)
    scores = cross_validate(model, X=df_train_base, y=df_train_y, scoring='r2', cv=kf)
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, timeout=60)

best_params = study.best_params
best_params_xgb_stac = best_params

kfolds = KFold(n_splits=4, shuffle=True, random_state=1)

model_xgb = xgb.XGBRegressor(**best_params_xgb_stac)
model_xgb.fit(df_train_base, df_train_y)

scores_xgb_stac = cross_validate(model_xgb, X=df_train_base, y=df_train_y, scoring='neg_mean_squared_error', cv=kfolds)
rmse_xgb_stac = np.sqrt(-scores_xgb_stac['test_score'].mean())


predict = model_xgb.predict(df_test_base)
predict = np.exp(predict) - 1

submittion = pd.DataFrame(data={'Id': df_test['Id'],
                                'SalePrice': predict})
submittion.to_csv('sub_HousePrice_Stacking.csv', index=None)

print('rmse_lgb', rmse_lgb)
print('rmse_xgb', rmse_xgb)
print('rmse_gbr', rmse_gbr)
print('rmse_knl', rmse_knl)
print('rmse_en', rmse_en)
print('rmse_xgb_stac:', rmse_xgb_stac)
