# Basic Library
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import scipy.stats as stats
import warnings
warnings.filterwarnings("ignore")

# sklearn utility
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics   
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedStratifiedKFold

## XGBoost
from xgboost import XGBClassifier
import xgboost as xgb

### LightGBM
from lightgbm import LGBMClassifier
import lightgbm as lgb

### CatBoost
from catboost import CatBoostClassifier
import catboost as catboost

## sklearn ensembles 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# Titanic Dataset
titanic_train = pd.read_csv("train.csv")
titanic_test = pd.read_csv("test.csv")
dataset = "titanic"
IdCol = 'PassengerId'
targetCol = 'Survived'
titanic_train.head()

int_or_float = titanic_train.dtypes[titanic_train.dtypes.isin(['int64', 'float64'])].index
print("Int or Flaot Columns : ", list(int_or_float))
num_cols = ['Age', 'SibSp', 'Parch', "Fare"]
print("Num Cols : ", num_cols)
cat_cols = ['Pclass', 'Sex', 'Embarked']
print("Cat Cols : ", cat_cols)
train_len = len(titanic_train)
combined =  pd.concat(objs=[titanic_train, titanic_test], axis=0).reset_index(drop=True)
#combined.tail()

def missing_values_details(df):
    total = df.isnull().sum()
    
    missing_df = pd.DataFrame({'count_missing': total}).reset_index().rename(columns={'index':'column_name'})
    missing_df['percent_missing'] = missing_df['count_missing']/len(df)
    missing_df = missing_df.sort_values(by='count_missing', ascending=False)
    missing_df = missing_df[missing_df['count_missing']!=0]
    print('Info : {} out of {} columns have mising values'.format(len(missing_df), len(df.columns)))
    missing_90 = missing_df[missing_df['percent_missing']>0.9]
    missing_80 = missing_df[missing_df['percent_missing']>0.8]
    missing_70 = missing_df[missing_df['percent_missing']>0.7]
    print("Info : {} columns have more that 90% missing values".format(len(missing_90)))
    print("Info : {} columns have more that 80% missing values".format(len(missing_80)))
    print("Info : {} columns have more that 70% missing values".format(len(missing_70)))
    
    return missing_df

def check_class_balance(df, target_col):
    counts = df[target_col].value_counts()
    class_df = pd.DataFrame(counts).reset_index().rename(columns={target_col:'counts', 'index':'class'})
    class_df.plot.bar(x='class', y='counts')
    print('Info : There are {} classes in the target column'.format(len(class_df)))
    max_class = class_df['counts'].max() 
    min_class = class_df['counts'].min()
    max_diff = max_class - min_class
    print("Info : Maximum difference between 2 classes is {} observations that is {} times w.r.t. minimum class".format(max_diff, (max_diff/min_class)))
    return class_df

def detect_outliers(df,n,features):
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from num_cols
outliers_rows = detect_outliers(titanic_train,2,num_cols)
print(len(outliers_rows))
# Drop outliers
titanic_train = titanic_train.drop(outliers_rows, axis = 0).reset_index(drop=True)

def describe_num_col(train, col):
    #### This function provides detailed comparison of a numerical varible
    ### missing value
    count_train = train[col].isnull().sum()
    #print("######    Variable Name : {}    ######".format(col))
    
    #### Skewness and Kurtosis
    train_k = stats.kurtosis(train[col].dropna(), bias=False)
    
    train_s = stats.skew(train[col].dropna(), bias=False)
    
    #### Outliers
    
    def count_outliers(df, col):
        mean_d = np.mean(df[col])
        std_d = np.std(df[col])
        
        scaled = (df[col]-mean_d)/std_d
        outliers = abs(scaled) > 3
        if len(outliers.value_counts()) > 1:
            return outliers.value_counts()[1]
        else:
            return 0   
    
    train_o = count_outliers(train, col)
        
    summ_df = pd.DataFrame({'info':['missing_count', 'missing_percent', 'skewness', 'kurtosis', 'outlier_count', 'outlier_percent'],
                           'train_set':[count_train, (count_train/len(train))*100, train_s, train_k, train_o, (train_o/len(train))*100]})
    
#     print("######    Summary Data")
#     display(summ_df)
    
    #print("######    Distribution and Outliers comparision plots")
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    plot10 = sns.distplot(train[train['Survived']==0][col],ax=ax1, label='Not Survived')
    sns.distplot(train[train['Survived']==1][col],ax=ax1,color='red', label='Survived')
    plot10.axes.legend()
    ax1.set_title('Distribution of {name}'.format(name=col))
    
    sns.boxplot(x='Survived',y=col,data=train,ax=ax2)
    #plt.xticks(ticks=[0,1],labels=['Non-Diabetes','Diabetes'])
    ax2.set_xlabel('Category') 
    ax2.set_title('Boxplot of {name}'.format(name=col))
    
    
    fig.show()    
    
    return
for col in num_cols:
    describe_num_col(titanic_train, col)
combined["Fare"] = combined["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

def describe_cat_col(df, col):
    ### unique values
    count_u = df[col].nunique()
    #print("Info : There are {} unique values".format(count_u))
    nulls = df[col].isnull().sum()
    #print("Info : There are {} missing values that is {} percent".format(nulls, nulls/len(df)))
    
    ### Percent share df
    share_df = pd.DataFrame(df[col].value_counts()).reset_index().rename(columns={'index':'class_name',col:'counts'})
    share_df['percent_share'] = share_df['counts']/sum(share_df['counts'])
    share_df = share_df.sort_values(by='percent_share', ascending=False)
    #display(share_df)
        
        
    if (count_u > 3 and count_u < 10):
        fig, ax  = plt.subplots()
        fig.suptitle(col + ' Distribution', color = 'red')
        explode = list((np.array(list(df[col].dropna().value_counts()))/sum(list(df[col].dropna().value_counts())))[::-1])
        labels = list(df[col].dropna().unique())
        sizes = df[col].value_counts()
        #ax.pie(sizes, explode=explode, colors=bo, startangle=60, labels=labels,autopct='%1.0f%%', pctdistance=0.9)
        ax.pie(sizes,  explode=explode, startangle=60, labels=labels,autopct='%1.0f%%', pctdistance=0.9)
        ax.add_artist(plt.Circle((0,0),0.2,fc='white'))
        plt.show()
    
    else:
        plt.figure()
        plt.title(col + ' Distribution', color = 'red')
        sns.barplot(x=col,y='Survived', data = df)
        plt.show()
        
    return

for col in cat_cols:
    #print("Column Name : {}".format(col))
    describe_cat_col(titanic_train, col)


combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].value_counts().index[0])
combined['Embarked'].isnull().sum()

combined = pd.get_dummies(combined, columns = ["Embarked"], prefix="Em")
# Explore Age vs Sex, Parch , Pclaß
# convert Sex into categorical value 0 for male and 1 for female
combined["Sex"] = combined["Sex"].map({"male": 0, "female":1})

# Filling missing value of Age 

## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp
# Index of NaN age rows
index_NaN_age = list(combined["Age"][combined["Age"].isnull()].index)

for i in index_NaN_age :
    age_med = combined["Age"].median()
    age_pred = combined["Age"][((combined['SibSp'] == combined.iloc[i]["SibSp"]) & (combined['Parch'] == combined.iloc[i]["Parch"]) & (combined['Pclass'] == combined.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred) :
        combined['Age'].iloc[i] = age_pred
    else :
        combined['Age'].iloc[i] = age_med

combined['Cabin'].describe()

combined["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in combined['Cabin'] ])
combined = pd.get_dummies(combined, columns = ["Cabin"],prefix="Cabin")

combined_title = [i.split(",")[1].split(".")[0].strip() for i in combined["Name"]]
combined["Title"] = pd.Series(combined_title)
combined["Title"].head()

# Convert to categorical values Title 
combined["Title"] = combined["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined["Title"] = combined["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
combined["Title"] = combined["Title"].astype(int)


# Drop Name variable
combined.drop(labels = ["Name"], axis = 1, inplace = True)
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(combined.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
combined["Ticket"] = Ticket
combined["Ticket"].head()

combined = pd.get_dummies(combined, columns = ["Ticket"], prefix="T")

# Create categorical values for Pclass
combined["Pclass"] = combined["Pclass"].astype("category")
combined = pd.get_dummies(combined, columns = ["Pclass"],prefix="Pc")
# Drop useless variables 
combined.drop(labels = ["PassengerId"], axis = 1, inplace = True)

train = combined[:train_len]
test = combined[train_len:]
test.drop(labels=["Survived"],axis = 1,inplace=True)
train["Survived"] = train["Survived"].astype(int)
y = train["Survived"]
train = train.drop(labels = ["Survived"],axis = 1)
train_x, val_x, train_y, val_y = train_test_split(train, y, test_size=0.2)
test_id = titanic_test['PassengerId']
test_x = test

assert len(train_x.columns) == len(test_x.columns)
### Function to evaluate model performance

def evaluate_model_performnce(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizing model performance
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 

    tn, fp, fn, tp = cm.ravel()
    #print(tn, fp, fn, tp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = ((tp+tn)/(tp+tn+fp+fn))*100
    print("Precision : ",precision)
    print("Recall : ",recall)
    print("F1 Score : ",f1)
    print("Validation Accuracy : ",accuracy)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy Score : ", accuracy)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    auc = metrics.auc(fpr, tpr)
    print("AUC Value : ", auc)
    
    return accuracy, auc, f1
## Function to create the submission file

def make_submission_file(filename, probab, test_id, IdCol, targetCol, threshold=None):
    submit = pd.DataFrame()
    submit[IdCol] = test_id
    submit[targetCol] = probab
    if threshold!=None:
        pred = [1 if x>=threshold else 0 for x in probab]
        submit[targetCol] = pred
    submit.to_csv(filename, index=False)
    return submit

xgb_clf = XGBClassifier()
xgb_clf.fit(train_x, train_y,eval_metric=["auc", "logloss"],verbose=True)

threshold = 0.4
xgb_val_prob = xgb_clf.predict_proba(val_x)
xgb_val_prob = pd.DataFrame(xgb_val_prob)[1]
xgb_val_pred = [1 if x >= threshold else 0 for x in xgb_val_prob]
xgb_acc, xgb_auc, xgb_f1 = evaluate_model_performnce(val_y, xgb_val_pred) 
xgb_prob = xgb_clf.predict_proba(test_x)
xgb_prob = pd.DataFrame(xgb_prob)[1]
xgb_sub = make_submission_file(dataset+"_xgb_default.csv", xgb_prob, test_id, IdCol, targetCol, threshold=0.5)
xgb_sub.head()

lgb_clf = LGBMClassifier()
lgb_clf.fit(train_x, train_y)

threshold = 0.4
lgb_val_prob = lgb_clf.predict_proba(val_x)
lgb_val_prob = pd.DataFrame(lgb_val_prob)[1]
lgb_val_pred = [1 if x >= threshold else 0 for x in lgb_val_prob]
lgb_acc, lgb_auc, lgb_f1 = evaluate_model_performnce(val_y, lgb_val_pred)

lgb_prob = lgb_clf.predict_proba(test_x)
lgb_prob = pd.DataFrame(lgb_prob)[1]

lgb_sub = make_submission_file(dataset+"_lgb_default.csv", lgb_prob, test_id, IdCol, targetCol, threshold=0.5)
lgb_sub.head()

cat_clf = CatBoostClassifier(verbose=0)
cat_clf.fit(train_x, train_y)


threshold = 0.4
cat_val_prob = cat_clf.predict_proba(val_x)
cat_val_prob = pd.DataFrame(cat_val_prob)[1]
cat_val_pred = [1 if x >= threshold else 0 for x in cat_val_prob]
cat_acc, cat_auc, cat_f1 = evaluate_model_performnce(val_y, cat_val_pred)

cat_prob = cat_clf.predict_proba(test_x)
cat_prob = pd.DataFrame(cat_prob)[1]

cat_sub = make_submission_file(dataset+"_cat_default.csv", cat_prob, test_id, IdCol, targetCol, threshold=0.5)
cat_sub.head()

ens_val_prob = 0.4*cat_val_prob + 0.3*lgb_val_prob + 0.3*xgb_val_prob
threshold = 0.4
ens_val_pred = [1 if x >= threshold else 0 for x in ens_val_prob]
ens_acc, ens_auc, ens_f1 = evaluate_model_performnce(val_y, ens_val_pred) 

ens_prob = 0.4*cat_prob + 0.3*lgb_prob + 0.3*xgb_prob
#y_prob
ens_sub = make_submission_file(dataset+"_weighted_ens.csv", ens_prob, test_id, IdCol, targetCol, threshold=0.5)
ens_sub.head()