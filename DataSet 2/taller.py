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