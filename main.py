import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib.testing import param
from seaborn import categorical
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 1000)
df = pd.read_csv('/Users/ertugrul/PycharmProjects/DataScienceMLCourse/RandomForest/14-income_evaluation.csv')

df.head()
df.shape

df.columns

col_names = ['age', 'workclass', 'finalweight', 'education', 'education_num', 'marital_status',
             'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
             'hours_per_week', 'native_country', 'income']

df.columns = col_names

df.columns

df.info()

df.describe()

categorical = [col for col in df.columns if df[col].dtype == 'str']
numerical = [col for col in df.columns if df[col].dtype != 'str']

categorical
numerical

df[categorical].head()

for col in categorical:
    print(df[col].value_counts())

fig, ax = plt.subplots(figsize=(8, 6))

ax = sns.countplot(x= "income", data = df, hue= "sex")
ax.set_title("Distribution of income with gender")

ax = sns.countplot(x= "income", data = df, hue= "race")
ax.set_title("Distribution of income with race")

sns.catplot(x= df["hours_per_week"], hue= df["income"])

df['workclass'].value_counts()
df['workclass'].unique()
df['workclass'] = df['workclass'].replace(" ?", np.nan)
df['workclass'].value_counts()

categorical

df['occupation'].unique()

df['occupation'] = df['occupation'].replace(" ?", np.nan)


df['native_country'].unique()

df['native_country'] = df['native_country'].replace(" ?", np.nan)

df.isnull().sum()


X = df.drop('income', axis = 1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

for i in [X_train, X_test]:
    i['workclass'] = i['workclass'].fillna(X_train['workclass'].mode()[0])
    i['occupation'] = i['occupation'].fillna(X_train['occupation'].mode()[0])
    i['native_country'] = i['native_country'].fillna(X_train['native_country'].mode()[0])


X_train.isnull().sum()
X_test.isnull().sum()

y_train_binary = y_train.apply(lambda x: 1 if x.strip() == '>50K' else 0)
target_means = y_train_binary.groupby(X_train['native_country']).mean()

X_train['native_country_encoded'] = X_train['native_country'].map(target_means)
X_train['native_country_encoded'] = X_train['native_country_encoded'].fillna(y_train_binary.mean())

X_test['native_country_encoded'] = X_test['native_country'].map(target_means)
X_test['native_country_encoded'] = X_test['native_country_encoded'].fillna(y_train_binary.mean())

X_train = X_train.drop('native_country', axis = 1)
X_test = X_test.drop('native_country', axis = 1)

one_hot_categories = [
    "workclass", "education", "marital_status",
    "occupation", "relationship", "race",
    "sex"
]

encoder = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown= "ignore", sparse_output= False), one_hot_categories),
    ],
    remainder='passthrough'
)

X_train_enc = encoder.fit_transform(X_train)
X_test_enc = encoder.transform(X_test)

X_train_enc
columns = encoder.get_feature_names_out()
X_train = pd.DataFrame(X_train_enc, columns=columns, index= X_train.index)
X_test = pd.DataFrame(X_test_enc, columns=columns, index= X_test.index)

X_train.head()


cols = X_train.columns

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=cols)
X_test = pd.DataFrame(X_test, columns=cols)


rfc = RandomForestClassifier(n_estimators= 100, random_state= 15)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

feature_scores = pd.Series(rfc.feature_importances_, index = X_train.columns).sort_values(ascending = False)
feature_scores.tail(10)


X_train = X_train.drop(['cat__education_ 12th', 'cat__race_ Other', 'cat__education_ 5th-6th', 'cat__education_ 1st-4th', 'cat__marital_status_ Married-AF-spouse', 'cat__occupation_ Priv-house-serv', 'cat__workclass_ Without-pay', 'cat__education_ Preschool',  'cat__occupation_ Armed-Forces', 'cat__workclass_ Never-worked'], axis = 1)
X_test = X_test.drop(['cat__education_ 12th', 'cat__race_ Other', 'cat__education_ 5th-6th', 'cat__education_ 1st-4th', 'cat__marital_status_ Married-AF-spouse', 'cat__occupation_ Priv-house-serv', 'cat__workclass_ Without-pay', 'cat__education_ Preschool',  'cat__occupation_ Armed-Forces', 'cat__workclass_ Never-worked'], axis = 1)

rfc = RandomForestClassifier(n_estimators= 100, random_state= 15)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


rf_params ={
    "n_estimators": [100, 200, 500, 1000],
    "max_depth": [5, 8, 10, 15, None],
    "max_features": ["sqrt", "log2", 5, 6, 7, 8],
    "min_samples_split" : [2, 8, 15, 20]
}
rfc = RandomForestClassifier()

rscv = RandomizedSearchCV(estimator= rfc, param_distributions= rf_params, cv= 3, n_jobs= -1)
rscv.fit(X_train, y_train)
y_pred = rscv.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

rscv.best_params_