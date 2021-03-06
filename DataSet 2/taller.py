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

# Validacion cruzada

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


print('*'*50)
print('Regresi??n Log??stica')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {logreg.score(x_test_out, y_test_out)}')

# Matriz de confusi??n
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

svc = SVC(gamma='auto')

svc.fit(x_train, y_train)

arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

kfold = KFold(n_splits=10)

acc_scores_train_train = []
acc_scores_test_train = []
# M??TRICAS

for train, test in kfold.split(x, y):
    arbol.fit(x[train], y[train])
    scores_train_train = arbol.score(x[train], y[train])
    scores_test_train = arbol.score(x[test], y[test])
    acc_scores_train_train.append(scores_train_train)
    acc_scores_test_train.append(scores_test_train)
    
y_pred = arbol.predict(x_test_out)


print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {svc.score(x_test_out, y_test_out)}')

arbol = DecisionTreeClassifier()

arbol.fit(x_train, y_train)


print('*'*50)
print('Decisi??n Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {arbol.score(x_test_out, y_test_out)}')

forest = RandomForestClassifier()

forest.fit(x_train, y_train)

print('*'*50)
print('RANDOM FOREST')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {forest.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {forest.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {forest.score(x_test_out, y_test_out)}')

nayve = GaussianNB()

nayve.fit(x_train, y_train)

print('*'*50)
print('NAYVE BAYES')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {nayve.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {nayve.score(x_test, y_test)}')

# Accuracy de Validaci??n
print(f'accuracy de Validaci??n: {nayve.score(x_test_out, y_test_out)}')

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

# ENTRENAMIENTO

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