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
data.job.replace(['student','admin.','retired','management','services','technician','self-employed','entrepreneur','unemployed','housemaid','blue-collar','unknown'], 
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


data.drop(['job', 'marital', 'balance', 'day','month','duration','pdays'], axis= 1, inplace = True)

data_train = data[:30000]
data_test = data[30000:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)


# validacion cruzada

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

for train, test in kfold.split(x, y):
    logreg.fit(x[train], y[train])
    scores_train_train = logreg.score(x[train], y[train])
    scores_test_train = logreg.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = logreg.predict(x_test_out)

print('*'*50)
print('Regresi??n Log??stica Validaci??n cruzada')



# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {logreg.score(x_test_out, y_test_out)}')


print(f'Matriz de confusi??n: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusion regresion logistica")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisi??n: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')


arbol = DecisionTreeClassifier()

# Entreno Del modelo
arbol.fit(x_train, y_train)

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []

for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = arbol.predict(x_test_out)

{}

forest = RandomForestClassifier()

# Entreno el modelo
forest.fit(x_train, y_train)

# M??TRICAS

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
# M??TRICAS

for train, test in kfold.split(x, y):
    forest.fit(x[train], y[train])
    scores_train_train = forest.score(x[train], y[train])
    scores_test_train = forest.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = forest.predict(x_test_out)

print('*'*50)
print('Random forest con Validaci??n cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {forest.score(x_test_out, y_test_out)}')


# Matriz de confusi??n
print(f'Matriz de confusi??n: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusion Random forest")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisi??n: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')

# Nuevo modelo

nayve = GaussianNB()

# Entreno el modelo
nayve.fit(x_train, y_train)

# M??TRICAS

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
# M??TRICAS

for train, test in kfold.split(x, y):
    nayve.fit(x[train], y[train])
    scores_train_train = nayve.score(x[train], y[train])
    scores_test_train = nayve.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = nayve.predict(x_test_out)


print('*'*50)
print('Nayve bayes Validaci??n cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {nayve.score(x_test_out, y_test_out)}')


# Matriz de confusi??n
print(f'Matriz de confusi??n: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusion Nayve")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisi??n: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')


{}

svc = SVC(gamma='auto')

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []


for train, test in kfold.split(x, y):
    svc.fit(x[train], y[train])
    scores_train_train = svc.score(x[train], y[train])
    scores_test_train = svc.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = svc.predict(x_test_out)


print('*'*50)
print('Maquina de soporte vectorial con Validaci??n cruzada')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {np.array(acc_scores_train_train).mean()}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {np.array(acc_scores_test_train).mean()}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {svc.score(x_test_out, y_test_out)}')


# Matriz de confusi??n
print(f'Matriz de confusi??n: {confusion_matrix(y_test_out, y_pred)}')

matriz_confusion = confusion_matrix(y_test_out, y_pred)
plt.figure(figsize = (6, 6))
sns.heatmap(matriz_confusion)
plt.title("Matriz de confusion maquina de soporte vectorial")

precision = precision_score(y_test_out, y_pred, average=None).mean()
print(f'Precisi??n: {precision}')

recall = recall_score(y_test_out, y_pred, average=None).mean()
print(f'Re-call: {recall}')

f1_score = f1_score(y_test_out, y_pred, average=None).mean()

print(f'f1: {f1_score}')
