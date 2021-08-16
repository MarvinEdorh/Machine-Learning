############################################### Costumer Lifetime Value ##################################################
import os; os.chdir('C:/Users/marvin/Python')

#################################################### SQL BigQuery ########################################################

import numpy as np ; import pandas as pd ; from google.cloud import bigquery

client = bigquery.Client.from_service_account_json(
json_credentials_path='mrvtestproject45-bbc9aec8eae9.json', 
project='mrvtestproject45')

#CA, lifetime et lifetime value de chaque consommateur en 2016
query = """
WITH lifetimevalue AS (
SELECT CONCAT('ID',fullvisitorid) AS ID_Visitor, IFNULL(SUM(hits.transaction.transactionRevenue/1000000),0) AS CA,  
       DATE_DIFF(PARSE_DATE('%Y%m%d',MAX(date)),PARSE_DATE('%Y%m%d',MIN(date)),DAY) AS lifetime
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_2016*` AS ga, UNNEST(ga.hits) AS hits
WHERE hits.transaction.transactionId IS NOT NULL GROUP BY fullvisitorid  )
SELECT ID_Visitor, CA, lifetime, CA*lifetime AS lifetimevalue FROM lifetimevalue ORDER BY CA DESC"""

query_results = client.query(query) ; query_results = query_results.result()

#Résutats de la reqête
ID_Visitor = [] ; CA = [] ; lifetime = [] ; lifetime_value = []

for row in query_results: 
    ID_Visitor.append(row[0]) 
    CA.append(row[1])
    lifetime.append(row[2])
    lifetime_value.append(row[3])
    
clv = {"CA":CA,"lifetime":lifetime,"lifetime_value":lifetime_value} ; clv = pd.DataFrame(clv, index = ID_Visitor)

################################################# Régression Linéaire ####################################################

import pandas as pd ; import numpy as np ; from sklearn.linear_model import LinearRegression 
from scipy import stats ; import statsmodels.api as sm

#créer Y et X
Y=clv['lifetime_value']
X=pd.DataFrame(np.c_[clv['CA'], clv['lifetime']], columns =['CA','lifetime'], index=ID_Visitor) 

#Méthode 1
model_Reg_lin_1 = LinearRegression() ; model_Reg_lin_1.fit(X,Y) ; R_sq = [model_Reg_lin_1.score(X,Y),'',''] 
coeff = model_Reg_lin_1.coef_ ; coeff = coeff.tolist() ; intercept = [] ; intercept.append(model_Reg_lin_1.intercept_) 
coeff = intercept + coeff ; pd.DataFrame({'coefficients' : coeff,'R²' : R_sq}, index =['intercept','CA','lifetime'])  
#le R2 est de 0,5 le modèle est assez significatif

#Méthode 2 avec p-valeur
X = sm.add_constant(X) # On ajoute l'intercept
model_Reg_lin_2 = sm.OLS(y, X) ; results = model_Reg_lin_2.fit() ; results.summary()

#################################################### prediction ##########################################################

#CA et lifetime des consommateurs de 2017
query = """
SELECT CONCAT('ID',fullvisitorid) AS ID_Visitor, IFNULL(SUM(hits.transaction.transactionRevenue/1000000),0) AS CA,  
       DATE_DIFF(PARSE_DATE('%Y%m%d',MAX(date)),PARSE_DATE('%Y%m%d',MIN(date)),DAY) AS lifetime
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_2017*` AS ga, UNNEST(ga.hits) AS hits
WHERE hits.transaction.transactionId IS NOT NULL GROUP BY fullvisitorid ORDER BY CA DESC"""

query_results = client.query(query) ; query_results = query_results.result()

#Résutats de la reqête
ID_Visitor = [] ; CA = [] ; lifetime = [] 

for row in query_results: 
    ID_Visitor.append(row[0]) 
    CA.append(row[1])
    lifetime.append(row[2])
    
clv_predict = {"CA":CA,"lifetime":lifetime} ; clv_predict = pd.DataFrame(clv_predict, index = ID_Visitor)

lifetime_value_predict = model_Reg_lin_1.predict(clv_predict) ; lifetime_value_predict = lifetime_value_predict.tolist() 
clv_predict['lifetime_value_predict'] = lifetime_value_predict
