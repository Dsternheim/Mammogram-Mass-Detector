"""
    Title: Mammogram Mass Detector
    Author: David Sternheim

    Description:
        The purpose of this script is to take data regarding mass detected in a mammogram and use machine learning
        models to predict if this mass is malignant or benign. The data is taken form UCI public data sets.

        Breakdown of the data set:
            The data has 961 instances of masses detected in mammograms. It's stored in mammographic_masses.data.txt.
            The format of the file is comma separated values with each of the following as one fo the values in order:
                1. BI-RADS Assessment: 1 to 5 (ordinal)
                2. Age: patient's age in years (integer)
                3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
                4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
                5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
                6. Severity: benign=0 malignant=1
            NOTE: '?' denotes a missing data value
    Last Updated: 09/15/18

    Known Bugs:

"""

import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression

"""
    Reading in the data and pre-processing it.
"""

data = pd.read_csv('Assets/mammographic_masses.data.txt')
df = pd.DataFrame(data)
df.columns = ['BIRADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
print(df.head())
d = {'1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0, '5': 5.0, '?': -1.0}
df['BIRADS'] = df['BIRADS'].map(d)
df['Shape'] = df['Shape'].map(d)
df['Margin'] = df['Margin'].map(d)
df['Density'] = df['Density'].map(d)
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Severity'] = pd.to_numeric(df['Severity'], errors='coerce')
df.fillna(-1.0, inplace=True)
df = df.astype('float32')
print(type(df['Severity'][0]))
"""
    Implement Decision Tree. Trained with K-Folds Cross Validation with K=10
"""
y = df['Severity']
features = list(df.columns[:5])
x = df[features]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.4, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
scores = model_selection.cross_val_score(clf, x, y, cv=10)
print('Decision Tree accuracy: ' + str(round(scores.mean()*100, 2)) + '%')  # ~76% accuracy

# Random Forests
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(x_train, y_train)
score = clf.score(x_test, y_test)
scores = model_selection.cross_val_score(clf, x, y, cv=10)
print('Random Forest accuracy: ' + str(round(scores.mean()*100, 2)) + '%')  # ~78% accuracy

"""
    Implement K-Nearest Neighbors. Trained with K-Folds Cross validation with K=10
"""
scaler = StandardScaler()
scaler = scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
clf = KNeighborsClassifier(n_neighbors=5)
clf = clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)
scores = model_selection.cross_val_score(clf, x, y, cv=10)
print('K-Nearest Neighbor accuracy: ' + str(round(scores.mean()*100, 2)) + '%')  # ~79%

"""
    Implement Naive Bayes. Trained with K-Folds Cross Validation with K=10
"""
clf = GaussianNB()
clf = clf.fit(x_train, y_train)
scores = model_selection.cross_val_score(clf, x, y, cv=10)
print('Naive Bayes accuracy: ' + str(round(scores.mean()*100, 2)) + '%')  # ~78%

"""
    Implement Support Vector Machine
"""
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(x_train, y_train)
scores = model_selection.cross_val_score(svc, x, y, cv=10)
print('Support Vector Machine accuracy: ' + str(round(scores.mean()*100, 2)) + '%')  # ~79%

"""
    Implement Logistic Regression. Trained with K-Folds Cross Validation.
"""
lgr = LogisticRegression()
lgr = lgr.fit(x_train, y_train)
scores = model_selection.cross_val_score(lgr, x, y, cv=10)
print('Logistic Regression accuracy: ' + str(round(scores.mean()*100, 2)) + '%')  # ~79%

"""
    Conclusions: Most machine learning models have an accuracy around 79%. DecisionTrees are by far the worst model to 
    detect if mass is malignant or benign because test returned a result of around 76%. Any of the other test can be 
    used to relative accuracy ~79%. The highest accuracy came from KNN at a high 79%. By adjusting hyper parameters, the
    models may be improved.
"""

