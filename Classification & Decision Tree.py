############################## Machine Learning Classification & Desisions Tree #####################################

import os; os.chdir('C:/Users/marvin/Desktop/Python')

#On modélise le phénomene d'achat afin de de predire quels visiteurs sont susceptibles d'efectuer une transaction

################################################### Train ##########################################################

import pandas as pd; import numpy as np ; transactions = pd.read_csv('transactions.csv', sep=",")
#Caractéritiques de chaque transaction : device de l'achat, systeme d'exploitation, source campaign, source trafic, 
#continent, produits et catégorie produit, nombre de visites sur produits et catégorie produit achetés avant achat

#le modele de classsification s'applique avec des variables numériques
#il faut recoder les variables catégorielles en effectuant un encodage one hot, chaque modalité devient une variable 
#qui prend 1 si l'individu la possede 0 sinon

col = list(transactions.columns); del col[0];del col[7];del col[7]

transactions_cat = pd.DataFrame(np.c_[transactions.iloc[:,1:8],transactions.iloc[:,[10]]], columns = col) 

transactions_cat_bis = pd.get_dummies(transactions_cat[transactions_cat.columns[:-1]])

#On y ajoute les variables numériques

col = list(transactions_cat_bis.columns) ; 
col.append('Products_Visits');col.append('Products_Category_Visits');col.append('transaction')
transactions_class = pd.DataFrame(np.c_[transactions_cat_bis.iloc[:,0:391],transactions.iloc[:,8:10],
                                      transactions.iloc[:,[10]]], columns = col) 

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
#1-recall = tx de faux pos ou tx de faux neg (ici 0% de faux neg mais 50% de faux pos)

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

##################################################  Prediction  ######################################################

future_transac = pd.read_csv('future_transactions_1.csv', sep=",")
col = list(future_transac.columns)              
del col[0];del col[7];del col[7]

future_transac_cat = pd.DataFrame(np.c_[future_transac.iloc[:,1:8]], 
                                        columns = col, index = future_transac['fullvisitorid']) 

#encodage one hot des variable catégorielles
future_transac_cat_bis = pd.get_dummies(future_transac_cat)

#On y ajoute les variables numériques
col = list(future_transac_cat_bis.columns)              
col.append('Products_Visits');col.append('Products_Category_Visits');
future_transac_class = pd.DataFrame(np.c_[future_transac_cat_bis.iloc[:,0:306], future_transac.iloc[:,8:10],
                                ], columns = col, index = future_transac['fullvisitorid'])

#On qpplique le modele pour prédire quels visiteurs sont le plus suceptibles d'effectuer une transaction
future_transac_class['CA_predit'] = pd.DataFrame(arbreFirst.predict(future_transac_class)) 
