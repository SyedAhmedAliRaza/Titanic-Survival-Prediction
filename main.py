import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

data = sns.load_dataset('titanic')

print("Titanic data:")
print(data)

print("Missing values in data:")
print(data.isnull().sum())

data['age'] = data['age'].fillna(data['age'].mean())
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])

data['family'] = data['sibsp'] + data['parch'] + 1
data['alone'] = 0
data.loc[data['family'] == 1, 'alone'] = 1

data['age_group'] = pd.cut(data['age'], bins=[0,12,18,30,50,100], labels=[0,1,2,3,4])
data['age_group'] = data['age_group'].astype(int)

data['fare_group'] = pd.qcut(data['fare'], q=4, labels=[0,1,2,3])
data['fare_group'] = data['fare_group'].astype(int)

data['sibsp_group'] = pd.cut(data['sibsp'], bins=[-1,0,1,2,8], labels=[0,1,2,3])
data['sibsp_group'] = data['sibsp_group'].astype(int)

data['parch_group'] = pd.cut(data['parch'], bins=[-1,0,1,2,6], labels=[0,1,2,3])
data['parch_group'] = data['parch_group'].astype(int)

data = data.drop(['deck','embark_town','alive','who','adult_male','class'], axis=1)

le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['embarked'] = le.fit_transform(data['embarked'])

X = data[['pclass', 'sex', 'age_group', 'sibsp_group', 'parch_group', 'fare_group', 'embarked', 'family', 'alone']]
y = data['survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, class_weight='balanced', random_state=42)
tree.fit(x_train, y_train)

y_pred = tree.predict(x_test)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print("Accuracy:", acc)
print("Recall:", rec)

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Titanic Prediction')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

features = ['pclass', 'sex', 'sibsp_group', 'parch_group', 'embarked', 'family', 'alone', 'age_group', 'fare_group']
for f in features:
    plt.figure()
    sns.barplot(x=f, y='survived', data=data)
    plt.title('Survival by ' + f)
    plt.xlabel(f)
    plt.ylabel('Survival Rate')
    plt.show()

sparse = ['sibsp_group', 'parch_group', 'family']
for f in sparse:
    plt.figure()
    sns.countplot(x=f, hue='survived', data=data)
    plt.title('Survivors by ' + f)
    plt.xlabel(f)
    plt.ylabel('Count')
    plt.legend(['Not Survived', 'Survived'])
    plt.show()

plt.figure()
sns.boxplot(x='survived', y='age', data=data)
plt.title('Age vs Survival')
plt.xlabel('Survived')
plt.ylabel('Age')
plt.show()

plt.figure()
sns.boxplot(x='survived', y='fare', data=data)
plt.title('Fare vs Survival')
plt.xlabel('Survived')
plt.ylabel('Fare')
plt.show()
