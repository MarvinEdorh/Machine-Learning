############################## Machine Learning Classification & Desisions Tree #####################################

import os; os.chdir('C:/Users/Desktop/Python')

#On modélise le phénomene d'achat afin de de predire quels visiteurs sont susceptibles d'efectuer une transaction

################################################# SQL BigQuery ######################################################

import numpy as np ; import pandas as pd ; from google.cloud import bigquery

client = bigquery.Client.from_service_account_json(
json_credentials_path='mrvtestproject45-bbc9aec8eae9.json', 
project='mrvtestproject45')

#Caractéritiques de chaque transaction : device de l'achat, systeme d'exploitation, source campaign, source trafic, 
#continent, produits et catégorie produit, nombre de visites sur produits et catégorie produit achetés avant achat

query = """
WITH 
transactions AS (
SELECT hits.transaction.transactionId AS ID_Transaction, device.deviceCategory,device.operatingSystem,
trafficSource.campaign, trafficSource.medium, geoNetwork.continent, fullvisitorid , hp.v2ProductName AS Product, 
hp.v2ProductCategory AS Product_Category,IFNULL(SUM(hits.transaction.transactionRevenue/1000000),0) AS CA, 
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20161201` AS ga, 
UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp 
GROUP BY ID_Transaction, device.deviceCategory,device.operatingSystem,trafficSource.campaign, 
trafficSource.medium, geoNetwork.continent, fullvisitorid , Product, Product_Category
ORDER BY CA DESC,ID_Transaction ), 
visits_products AS (  
SELECT fullvisitorid, hp.v2ProductName AS Product, SUM( totals.visits) AS Product_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,UNNEST(ga.hits) AS hits, 
UNNEST(hits.product) AS hp WHERE _TABLE_SUFFIX <= '20161201'
GROUP BY fullvisitorid, Product ORDER BY fullvisitorid, Product_Visits DESC ), 
visits_products_category AS (  
SELECT fullvisitorid, hp.v2ProductCategory AS Product_Category, SUM( totals.visits) Product_Category_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,UNNEST(ga.hits) AS hits, 
UNNEST(hits.product) AS hp WHERE _TABLE_SUFFIX <= '20161201'
GROUP BY fullvisitorid, Product_Category ORDER BY fullvisitorid, Product_Category_Visits DESC )
SELECT ID_Transaction, deviceCategory, operatingSystem, campaign, medium, continent,
transactions.Product, transactions.Product_Category, Product_Visits, Product_Category_Visits, 
CASE WHEN CA = 0 THEN 0 ELSE 1 END AS Transaction
FROM transactions LEFT JOIN visits_products
ON transactions.fullvisitorid = visits_products.fullvisitorid AND transactions.Product = visits_products.Product
LEFT JOIN visits_products_category
ON transactions.fullvisitorid = visits_products_category.fullvisitorid 
AND transactions.Product_Category = visits_products_category.Product_Category
ORDER BY CA DESC"""

query_results = client.query(query) ; query_results = query_results.result()

#Résutats de la reqête
ID_Transaction = [] ; deviceCategory = [] ;	operatingSystem	= [] ; campaign	= [] ; medium = [] ;	
continent = [] ; Product = [] ; Product_Category = [] ; Product_Visits = [] ; Product_Category_Visits = [] 
Transaction = [] ;

for row in query_results: 
    ID_Transaction.append(row[0]) 
    deviceCategory.append(row[1])
    operatingSystem.append(row[2])
    campaign.append(row[3])
    medium.append(row[4])
    continent.append(row[5])
    Product.append(row[6])
    Product_Category.append(row[7])
    Product_Visits.append(row[8])
    Product_Category_Visits	.append(row[9])
    Transaction.append(row[10])
       
transactions = {"ID_Transaction":ID_Transaction,
                  "deviceCategory":deviceCategory,
                  "operatingSystem":operatingSystem,
                  "campaign":campaign,
                  "medium":medium,
                  "continent":continent,
                  "Product":Product,
                  "Product_Category":Product_Category,
                  "Product_Visits":Product_Visits,
                  "Product_Category_Visits":Product_Category_Visits,
                  "Transaction":Transaction} 
transactions = pd.DataFrame(transactions)

######################################################## Train #########################################################

#le modele de classsification s'applique avec des variables numériques
#il faut recoder les variables catégorielles en effectuant un encodage one hot,
#chaque modalité devient une variable qui prend 1 si l'individu la possede 0 sinon

col = list(transactions.columns); del col[0]; del col[7]; del col[7]

transactions_cat = pd.DataFrame(np.c_[transactions.iloc[:,1:8],transactions.iloc[:,[10]]], columns = col) 

transactions_cat_one_hot = pd.get_dummies(transactions_cat[transactions_cat.columns[:-1]])

#Il doit y avoir exactement les mêmes modalités entre les données d'entrainement et celles de predictions.
#Dans l'anticipation de la prediction on ajoute les modalités qui pouraient ne pas de trouver dans le 
#jeu d'entrainement mais figurer celui de prediction (un produit jamais acheté mais figurant tout de meme sur
#le site, un visiteur d'un nouvau continent ou source de calpagne). On y applique ensuite également un encodage 
#one hot et elles prendront donc toutes la valeur 0 dans le jeu d´entrainement et 1 dans celui de prediction 
#si elles s'y trouvent.

modalites = pd.read_csv('segments.csv', sep=",") ; modalites_cat_one_hot = pd.get_dummies(modalites)  

col_1 = list(modalites_cat_one_hot.columns) ; col_2 = list(transactions_cat_one_hot.columns) 
#modalité totale ; #modalité présente

col_3 = [value for value in col_1 if value not in col_2] ; col_4 = col_2 + col_3
#modalité absente ; #modalité totale = #modalité présente + #modalité absente 

mod_add = pd.DataFrame( 0, columns = col_3, index=range(0,41222) ) 

#On y ajoute les variables numériques
col = col_4 ; col.append('Products_Visits') ; col.append('Products_Category_Visits') ; col.append('transaction')
transactions_class = pd.DataFrame(np.c_[transactions_cat_one_hot.iloc[:,0:416], mod_add.iloc[:,0:357],
                                        transactions.iloc[:,8:10], transactions.iloc[:,[10]]], columns = col) 

#on découpe la base en 2 avec la meme proprtion de modalité 
#sur la variable à modeliser dans la base train que dans la base test
from sklearn.model_selection import train_test_split
train, test = train_test_split(transactions_class, random_state=1, stratify = transactions_class.transaction)
print(train.transaction.value_counts(normalize=True)) ; print(test.transaction.value_counts(normalize=True))

#construction du modèle
from sklearn.tree import DecisionTreeClassifier ; arbreFirst = DecisionTreeClassifier()
X = train[train.columns[:-1]] ; y = train.transaction ; arbreFirst.fit(X, y)

#Evaluation du modele
eval_test = pd.DataFrame(arbreFirst.predict(test[train.columns[:-1]]))

#matrice de confusion
from sklearn import metrics ; metrics.confusion_matrix(test.transaction,eval_test )

print(metrics.classification_report(test.transaction,eval_test)) 
#1-recall = tx de faux pos ou tx de faux neg (ici 0% de faux negatifs mais 99% de faux positifs)

#taux de reconnaissance – (vrai positf + vrai négatif)/ effectif total 
metrics.accuracy_score(test.transaction,eval_test )

#taux d'erreur – (faux positf + faux négatif)/ effectif total
1.0 - metrics.accuracy_score(test.transaction,eval_test )

#sensibilité est la capacité du modèle à prédire un positif quand la donnée est réellement positive
#Faux positif - sensibilité (2eme ligne)
metrics.recall_score(test.transaction,eval_test,pos_label=1)

#spécificité est la capacité du modèle à prédire un négatif lorsqu'il y a vraiment un négatif.
#Faux négatif - spécificité 1ere ligne)
metrics.recall_score(test.transaction,eval_test ,pos_label=0)

#précision –  2eme colonne
metrics.precision_score(test.transaction,eval_test,pos_label=1)

#F1-score : moyenne harmonique entre rappel et précision :
metrics.f1_score(test.transaction,eval_test,pos_label=1)
    
#arbre de decicision
from sklearn.tree import plot_tree
plot_tree(arbreFirst,feature_names = list(X.columns),filled=True)

#affichage plus grand pour une meilleure lisibilité
import matplotlib.pyplot as plt
plt.figure(figsize=(150,200))
plot_tree(arbreFirst,feature_names = list(X.columns),filled=True)
plt.show()

#importance des variables
Importance_Var = pd.DataFrame({"Variable":X.columns,"Importance":arbreFirst.feature_importances_
                               }).sort_values(by="Importance",ascending=False) 

#On voit ici que la variable plus importante dans le fait d´effectuer une transaction est 
#le nombre de visites sur le produit acheté

#Méthode 2 xgboost

from xgboost import XGBClassifier ; from sklearn.preprocessing import MinMaxScaler ; import xgboost as xgb

#construction du modèle
boost = XGBClassifier() ; boost.fit(X, y)

#évaluation modèle
p_boost = boost.predict(X)
print ("Score Train -->", round(boost.score(X, y) *100,2), " %")

#importance des variables
xgb.plot_importance(boost)

#arbre de decicision
xgb.to_graphviz(boost, num_trees=2)

#####################################################  Prediction  ######################################################

query = """
WITH visitors AS (
SELECT DISTINCT fullvisitorid, device.deviceCategory, device.operatingSystem, 
trafficSource.campaign, trafficSource.medium, geoNetwork.continent,
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 
UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp 
WHERE _TABLE_SUFFIX BETWEEN '20161201' AND '20161231' ORDER BY fullvisitorid), 
visits_products AS (  
SELECT fullvisitorid, hp.v2ProductName AS Products, SUM(totals.visits) AS Products_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,
UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp 
WHERE _TABLE_SUFFIX BETWEEN '20161201' AND '20161231'
GROUP BY fullvisitorid, Products ORDER BY fullvisitorid, Products_Visits DESC ), 
visits_products_category AS (  
SELECT fullvisitorid, hp.v2ProductCategory AS Products_Category, SUM(totals.visits) AS Products_Category_Visits
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga,
UNNEST(ga.hits) AS hits, UNNEST(hits.product) AS hp 
WHERE _TABLE_SUFFIX BETWEEN '20161201' AND '20161231'
GROUP BY fullvisitorid, Products_Category ORDER BY fullvisitorid, Products_Category_Visits DESC )
SELECT visitors.fullvisitorid, deviceCategory, operatingSystem, campaign, medium, continent, 
Products, Products_Category,Products_Visits, Products_Category_Visits, FROM visitors
LEFT JOIN visits_products USING (fullvisitorid)
LEFT JOIN visits_products_category USING (fullvisitorid)"""

query_results = client.query(query) ; query_results = query_results.result()

#Résutats de la reqête
fullvisitorid = [] ; deviceCategory = [] ;	operatingSystem	= [] ; campaign	= [] ; medium = [] ;	
continent = [] ; Product = [] ; Product_Category = [] ; Product_Visits = [] ; Product_Category_Visits = [] 
Transaction = [] ;

for row in query_results: 
    fullvisitorid.append(row[0]) 
    deviceCategory.append(row[1])
    operatingSystem.append(row[2])
    campaign.append(row[3])
    medium.append(row[4])
    continent.append(row[5])
    Product.append(row[6])
    Product_Category.append(row[7])
    Product_Visits.append(row[8])
    Product_Category_Visits	.append(row[9])
       
future_transactions = {"fullvisitorid":fullvisitorid,
                       "deviceCategory":deviceCategory,
                       "operatingSystem":operatingSystem,
                       "campaign":campaign,
                       "medium":medium,
                       "continent":continent,
                       "Product":Product,
                       "Product_Category":Product_Category,
                       "Product_Visits":Product_Visits,
                       "Product_Category_Visits":Product_Category_Visits} 

future_transactions = pd.DataFrame(future_transactions)

col = list(future_transactions.columns) ; del col[0];del col[7];del col[7]

future_transactions_cat = pd.DataFrame(np.c_[future_transactions.iloc[:,1:8]], 
                                       columns = col, index = future_transactions['fullvisitorid']) 

#encodage one hot des variable catégorielles
future_transactions_cat_one_hot = pd.get_dummies(future_transactions_cat)

#On ajoute également les modalités qui ne figureraient pas dans le jeu de prediction mais qui étaient presentes 
#dans celui d´entrainement. Avec l´encodage one hot elles prendront cette fois toutes pour valeur 0.

col_1 = list(modalites_cat_one_hot.columns) ; col_2 = list(future_transactions_cat_one_hot.columns) 

col_3 = [value for value in col_1 if value not in col_2] ; col_4 = col_2 + col_3

mod_add = pd.DataFrame(0, columns = col_3, index=range(0,5680650))

#On y ajoute les variables numériques
col = col_4 ; col.append('Products_Visits') ; col.append('Products_Category_Visits') 
future_transactions_class = pd.DataFrame(np.c_[future_transactions_cat_one_hot.iloc[:,0:489],mod_add.iloc[:,0:284],
                                               future_transactions.iloc[:,8:10]], 
                                               columns = col, index = future_transactions['fullvisitorid'])

#On qpplique le modele pour prédire quels visiteurs sont le plus suceptibles d'effectuer une transaction
future_transactions_class['transactions_predict'] = arbreFirst.predict(future_transactions_class) 
