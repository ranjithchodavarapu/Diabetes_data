import pandas as pd
import sklearn
import numpy as np
from scipy.stats import ttest_1samp
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix,classification_report
data=pd.read_csv("datasets_228_482_diabetes.csv")

print(data.head())
print(data.info())
print(data.describe())
print(data.corr())
print(data['Age'].mean())
x=data.drop(['Outcome'],axis=1)
y=data['Outcome']

print(x.head())
print(y.head())
print(data['Age'].mean())
age=data['Age']
q1=np.percentile(age,25,interpolation='midpoint')
print(q1)
q2=np.percentile(age,50,interpolation='midpoint')
print(q2)
q3=np.percentile(age,75,interpolation='midpoint')
print(q3)

iqr=q3-q1
print(iqr)

ll=q1-1.5*iqr
ul=q3+1.5*iqr
print(ll)
print(ul)
tset,pval=ttest_1samp(age,33)
print(pval)
print(np.std(age))



model=ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
print(x_train,y_train)




cls=LogisticRegression()

cls.fit(x_train,y_train)
pre=cls.predict(x_test)

print(accuracy_score(y_test,pre))
print(precision_score(y_test,pre))
print(recall_score(y_test,pre))
print(f1_score(y_test,pre))
print(confusion_matrix(y_test,pre))
print(classification_report(y_test,pre))
