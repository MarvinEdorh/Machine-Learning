########################################### Machine Learning  ##########################################################

import os; os.chdir('C:/Users/marvi/Desktop/MsMDA/AutoFormation/Python')

########################################## Régression Linéaire #########################################

import pandas as pd ; transactions = pd.read_csv('transactions.csv', sep=",")
import numpy as np ; from sklearn.linear_model import LinearRegression 
from scipy import stats ; import statsmodels.api as sm

#créer y et X
y=transactions['CA']
X=pd.DataFrame(np.c_[transactions['Products'],transactions['Products_Category']], 
               columns =['Products','Products_Category']) 

X = sm.add_constant(X)

#Méthode 1
model_Reg_lin_1 = sm.OLS(y, X)
results = model_Reg_lin_1.fit()
print(results.summary())

#Méthode 2
model_Reg_lin_2 = LinearRegression()
X=pd.DataFrame(np.c_[transactions['Products'],transactions['Products_Category']], 
               columns =['Products','Products_Category'])  
model_Reg_lin_2.fit(X,y)
results_2 = model_Reg_lin_2.fit(X,y)
print(model_Reg_lin_2.score(X,y)) #R²
print(model_Reg_lin_2.intercept_) #intercept
print(model_Reg_lin_2.coef_)  #coefficient

#le R2 est de 0,1 le modele est peu significatif

#prédiction
model_Reg_lin_2.predict(X)
