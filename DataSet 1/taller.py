#IMPORT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import simplefilter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

url = 'bank-full.csv'
data = pd.read_csv(url)

# tratamiento de la data de bank-full

data.default.replace(['no','yes'], [0,1], inplace= True)
data.job.replace(['blue-collar','management','technician','admin.','services','retired','self-employed','entrepreneur','unemployed','housemaid','student','unknown'], 
[0,1,2,3,4,5,6,7,8,9,10,11], inplace= True)
data.marital.replace(['married','single','divorced'], [0,1,2], inplace= True)
data.education.replace(['secondary','tertiary','primary','unknown'], [0,1,2,3], inplace= True)
data.housing.replace(['no','yes'], [0,1], inplace= True)
data.loan.replace(['no','yes'], [0,1], inplace= True)
data.contact.replace(['cellular','unknown','telephone'], [0,1,2], inplace= True)
data.poutcome.replace(['unknown','failure','other','success'], [0,1,2,3], inplace= True)
data.y.replace(['no','yes'], [0,1], inplace= True)


rangos = [10,20,30,40,50]
nombres = ['1','2','3','4',]
data.age = pd.cut(data.age, rangos, labels=nombres)
data.dropna(axis=0,how='any', inplace=True)


