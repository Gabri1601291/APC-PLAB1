#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# Preparació de les heines per la pràctica

# In[92]:


from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import pyplot as plt
import scipy.stats

# Visualitzarem només 3 decimals per mostra
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Funcio per a llegir dades en format csv
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset


# Es treuen tots els atributs que siguin id, dates, objectes y aquells que siguin binaris.
# 

# Així doncs ara podreu respondre a les següents preguntes:
# 
# Quin és el tipus de cada atribut?  
# Quins atributs tenen una distribució Guassiana?  
# Quin és l'atribut objectiu? Per què?  

# In[93]:


#Carreguem les dades
dataset = load_dataset('HRDataset_v14.csv')


# In[94]:


#Mirem el tipus de cada atribut
print(dataset.dtypes)


# In[95]:


dataset.describe()


# In[96]:


print("Per comptar el nombre de valors no existents:")
print(dataset.isnull().sum())


# In[97]:


# treiem els atributs que no ens interesa
atrs = dataset.columns
del(dataset['Termd'])
# Eliminem els atributs ID ya que son irrellevants

ids = []
for atr in atrs:
    if(atr[len(atr)-1] == "D" and atr[len(atr)-2] == "I"):
        del(dataset[atr])

# Eliminem els atriibuts de tipus object ya que no son utils
objs = dataset.select_dtypes(include="object").columns
for obj in objs:
    del(dataset[obj])

# Tambe hem eliminat un atribut que era null 


# In[98]:


#Mirem el tipus de cada atribut
print(dataset.dtypes)


# In[99]:


dataset.describe()


# In[100]:


print("Per comptar el nombre de valors no existents:")
print(dataset.isnull().sum())


# In[101]:


# Guardem les dades en dues variables
dades = dataset.values
atrs = dataset.columns
x = dades[:,:]
y = dades[:,3]


# In[102]:


for i in range(x.shape[1]):
    plt.figure()
    plt.title("Histograma de l'atribut "+ atrs[i])
    plt.xlabel("Attribute Value")
    plt.ylabel("Count")
    hist = plt.hist(x[:,i], bins=20, range=[np.min(x[:,i]), np.max(x[:,i])], histtype="bar", rwidth=0.8)


# In[103]:


import seaborn as sns

# Mirem la correlació entre els atributs d'entrada per entendre millor les dades
correlacio = dataset.corr()

plt.figure()

ax = sns.heatmap(correlacio, annot=True, linewidths=.5)


# In[104]:


# Mirem la relació entre atributs utilitzant la funció pairplot
relacio = sns.pairplot(dataset)


# In[105]:


plt.figure()
ax = plt.scatter(x[:,2], y[:])


# # Apartat (B): Primeres regressions
# 
# Per a aquest primer apartat es calcularà l'error quadràtic mitjà només del regressor per a cada un dels atributs de la base de dades, determinant aquell atribut pel qual l'error quadràtic mitjà (entre el valor predit i el real, per a cada mostra) és més baix. 
# 
# A continuació se us dona una funció auxiliar per a calcular l'error quadràtic mitjà:

# In[106]:


import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def mean_squeared_error(y1, y2):
    # comprovem que y1 i y2 tenen la mateixa mida
    assert(len(y1) == len(y2))
    mse = 0
    for i in range(len(y1)):
        mse += (y1[i] - y2[i])**2
    return mse / len(y1)


def regression(x, y):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression()

    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)

    # Retornem el model entrenat
    return regr

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

# es pot utilitzar numpy per a calcular el mse
def mse(v1, v2):
    return ((v1 - v2)**2).mean()

def split_data(x, y, train_ratio=0.8):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    n_train = int(np.floor(x.shape[0]*train_ratio))
    indices_train = indices[:n_train]
    indices_val = indices[n_train:] 
    x_train = x[indices_train, :]
    y_train = y[indices_train]
    x_val = x[indices_val, :]
    y_val = y[indices_val]
    return x_train, y_train, x_val, y_val


# In[107]:


# Extraiem el primer atribut de x i canviem la mida a #exemples, #dimensions de l'atribut.
# En el vostre cas, haureu de triar un atribut com a y, i utilitzar la resta com a x.
atribut1 = x[:,2].reshape(x.shape[0], 1) 
regr = regression(atribut1, y) 
predicted = regr.predict(atribut1)

# Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
plt.figure()
ax = plt.scatter(x[:,2], y)
plt.plot(atribut1[:,0], predicted, 'r')

# Mostrem l'error (MSE i R2)
MSE = mse(y, predicted)
r2 = r2_score(y, predicted)

print("Mean squeared error: ", MSE)
print("R2 score: ", r2)


# In[108]:


x = dades[:,:]
y = dades[:,0]
x = standarize(x)

x_train, y_train, x_val, y_val = split_data(x, y)

y_train = y_train.reshape(y_train.shape[0], 1)

for i in range(1,len(atrs)):
    atribut1 = (x_train[:,i].reshape(x_train.shape[0], 1)) 
    regr = regression(atribut1, y_train) 
    predicted = regr.predict(atribut1)

    # Mostrem la predicció del model entrenat en color vermell a la Figura anterior 1
    
    plt.figure()
    plt.title(atrs[i])
    ax = plt.scatter(x[:,i], y)
    plt.plot(atribut1[:,0], predicted, 'r')
    
    # Mostrem l'error (MSE i R2)
    MSE = mse(y_train, predicted)
    r2 = r2_score(y_train, predicted)
    print("Salary vs " + atrs[i])
    print("Mean squeared error: ", MSE)
    print("R2 score: ", r2)
    print('')

