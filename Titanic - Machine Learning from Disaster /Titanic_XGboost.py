from multiprocessing import Pipe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------- データセットの読み込み --------
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

test_data['Survived'] = np.nan
df = pd.concat([train_data, test_data], ignore_index=True, sort=False)


sns.barplot(x='Sex', y='Survived', data=df, palette='Set3')
# plt.show()

# -------- Age --------
from sklearn.ensemble import RandomForestRegressor

age_df = df[['Age', 'Pclass', 'Sex', 'Parch', 'SibSp']]

age_df = pd.get_dummies(age_df)

known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values

X = known_age[:, 1:]
y = known_age[:, 0]

rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rfr.fit(X, y)

predictedAge = rfr.predict(unknown_age[:, 1:])

df.loc[df['Age'].isnull(), 'Age'] = predictedAge


# -------- Name --------
df['Title'] = df['Name'].map(lambda x: x.split(', ')[1].split('. ')[0])
df['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer', inplace=True)
df['Title'].replace(['Don', 'Sir',  'the Countess', 'Lady', 'Dona'], 'Royalty', inplace=True)
df['Title'].replace(['Mme', 'Ms'], 'Mrs', inplace=True)
df['Title'].replace(['Mlle'], 'Miss', inplace=True)
df['Title'].replace(['Jonkheer'], 'Master', inplace=True)

sns.barplot(x='Title', y='Survived', data=df, palette='Set3')
# plt.show()


# ------------ Surname ------------
df['Surname'] = df['Name'].map(lambda name:name.split(',')[0].strip())
df['FamilyGroup'] = df['Surname'].map(df['Surname'].value_counts()) 

Female_Child_Group = df.loc[(df['FamilyGroup']>=2) & ((df['Age']<=16) | (df['Sex']=='female'))]
Female_Child_Group = Female_Child_Group.groupby('Surname')['Survived'].mean()

Male_Adult_Group = df.loc[(df['FamilyGroup']>=2) & ((df['Age']>16) & (df['Sex']=='male'))]
Male_Adult_Group = Male_Adult_Group.groupby('Surname')['Survived'].mean()

Dead_list = set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
Survived_list = set(Male_Adult_Group[Male_Adult_Group.apply(lambda x:x==1)].index)

print('Dead_list = ', Dead_list)
print('Survived_list = ', Survived_list)

df.loc[(df['Survived'].isnull()) & (df['Surname'].apply(lambda x:x in Dead_list)),\
             ['Sex','Age','Title']] = ['male',28.0,'Mr']
df.loc[(df['Survived'].isnull()) & (df['Surname'].apply(lambda x:x in Survived_list)),\
             ['Sex','Age','Title']] = ['female',5.0,'Mrs']

# ------- Fare --------

fare = df.loc[(df['Embarked'] == 'S') & (df['Pclass'] == 3), 'Fare'].median()
df['Fare'] = df['Fare'].fillna(fare)


# -------- Family --------
df['Family'] = df['SibSp'] + df['Parch'] + 1
df.loc[(df['Family']>=2) & (df['Family']<=4), 'Family_label'] = 2
df.loc[(df['Family']>=5) & (df['Family']<=7) | (df['Family']==1), 'Family_label'] = 1 
df.loc[(df['Family']>=8), 'Family_label'] = 0
sns.barplot(x='Family', y='Survived', data=df, palette='Set3')
# plt.show()

sns.barplot(x='Pclass', y='Survived', data=df, palette='Set3')
# plt.show()


# -------- Ticket --------
Ticket_Count = dict(df['Ticket'].value_counts())
df['TicketGroup'] = df['Ticket'].map(Ticket_Count)

df.loc[(df['TicketGroup']>=2) & (df['TicketGroup']<=4), 'Ticket_label'] = 2
df.loc[(df['TicketGroup']>=5) & (df['TicketGroup']<=8) | (df['TicketGroup']==1), 'Ticket_label'] = 1  
df.loc[(df['TicketGroup']>=11), 'Ticket_label'] = 0
sns.barplot(x='Ticket_label', y='Survived', data=df, palette='Set3')
# plt.show()

# -------- Cabin --------
df['Cabin'] = df['Cabin'].fillna('Unknown')
df['Cabin_label'] = df['Cabin'].str.get(0)
sns.barplot(x='Cabin_label', y='Survived', data=df, palette='Set3')
#plt.show()

# -------- Embarked ---------
df['Embarked'] = df['Embarked'].fillna('S')
sns.barplot(x='Embarked', y='Survived', data=df, palette='Set3')
#plt.show()


# -------- 前処理 --------
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'Family_label', 'Ticket_label', 'Cabin_label']]

df = pd.get_dummies(df)

train = df[df['Survived'].notnull()]
test = df[df['Survived'].isnull()].drop('Survived', axis=1)

X = train.values[:, 1:]
y = train.values[:, 0]
test_X = test.values

# -------- モデル構築 --------
from xgboost import XGBClassifier
xgb_clf = XGBClassifier()
xgb_clf.fit(X, y, eval_metric=["auc", "logloss"], verbose=True)
threshold = 0.4
xgb_prob = xgb_clf.predict_proba(test_X)
xgb_prob = pd.DataFrame(xgb_prob)[1]
xgb_pred = [1 if x >= threshold else 0 for x in xgb_prob]


# -------- submit dataの作成 --------
PassengerId = test_data['PassengerId'].values

submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": xgb_pred})
submission.to_csv('Titanic_xgboost.csv', index=False)
