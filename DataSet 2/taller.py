from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

url = 'diabetes.csv'

data = pd.read_csv(url)

rangos = [1,5,10,15]
nombres = ['1','2','3']
data.Pregnancies = pd.cut(data.Pregnancies, rangos, labels=nombres)
data.Pregnancies.replace(np.nan, 1, inplace=True)

rangos2 = [0,10,30,50,100]
nombres2 = ['1','2','3','4']
data.Glucose = pd.cut(data.Glucose, rangos2, labels=nombres2)

rangos3 = [0,50,100,150]
nombres3 = ['1','2','3']
data.BloodPressure = pd.cut(data.BloodPressure, rangos3, labels=nombres3)

rangos4 = [0,100,200,300,400]
nombres4 = ['1','2','3','4']
data.Insulin = pd.cut(data.Insulin, rangos4, labels=nombres4)

rangos5 = [0,20,40,60]
nombres5 = ['1','2','3']
data.BMI = pd.cut(data.BMI, rangos5, labels=nombres5)


rangos6 = [5,10,15,20]
nombres6 = ['1','2','3']
data.Age = pd.cut(data.Age, rangos6, labels=nombres6)

data.dropna(axis=0,how='any', inplace=True)

data.drop(['SkinThickness','DiabetesPedigreeFunction'], axis= 1, inplace = True)

data_train = data[:450]
data_test = data[450:]

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome) 

logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

logreg.fit(x_train,y_train)

