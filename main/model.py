import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
import pickle

df = pd.read_csv('diabetes.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 50)

# initial data exploration

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# handling outliers

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
for idx, col in enumerate(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']):
    sns.boxplot(y=df[col], ax=axes[idx//4, idx%4])
plt.tight_layout()
plt.show()


for col in ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']:
    df = df[np.abs(df[col] - df[col].mean()) <= (3 * df[col].std())]

print(df.describe())

# scaling

scaler = StandardScaler()
df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']] = scaler.fit_transform(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']])

with open('scaler.pkl','wb') as f:
    pickle.dump(scaler,f)

# splitting the data

x = df.drop('Outcome',axis='columns')
y = df.Outcome

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# hyper parameter hyper tuning

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [1, 5, 10]
        }
    },
    'decision tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {
            'criterion': ['gini','entropy'],
            'splitter': ['best','random']
        }
    },
    'logistic regression': {
        'model': LogisticRegression(max_iter=1000,random_state=42),
        'params': {
            'C': [1,5,10]
        }
    }
}


# Evaluate each model using GridSearchCV
scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(x_train, y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

# Create a DataFrame from the scores
results_df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(results_df)

# svm the best model:

model = svm.SVC(C=1,kernel='linear',random_state=42)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

print(model.predict(x_test))

with open('model.pkl','wb') as f:
    pickle.dump(model,f)

# feature importance analysis for random forest model:

rf = RandomForestClassifier(n_estimators=10,random_state=42)
rf.fit(x_train, y_train)

feature_importance = pd.Series(rf.feature_importances_, index=x.columns)
feature_importance_sorted = feature_importance.sort_values(ascending=False)
print("Feature Importance:")
print(feature_importance_sorted)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance_sorted.index, feature_importance_sorted.values)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
