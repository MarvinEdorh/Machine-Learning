########################################### Macine Learning  ##########################################################

import os; os.chdir('C:/Users/marvin/Desktop/Python')

############################################## Classification #######################################################

import pandas as pd; import numpy as np ; transactions = pd.read_csv('transactions.csv', sep=",")

#on modélise les transactions avec un CA superieure à la moyenne
transactions['CA_fort'] = np.where(transactions['CA']>transactions['CA'].mean(),'oui','non') 

############################################ Variable Num ########################################################

#le modele de classsification s'applique avec des variables numériques
transactions_num = pd.DataFrame(np.c_[transactions.iloc[:,6:8],transactions.iloc[:,[9]]], 
                                columns = ['Products','Products_Category','CA_fort'],
                                index = transactions['ID_Transaction']) 

#on découpe la base en 2 avec la meme proprtion de modalité 
#sur la variable à modeliser dans la base train que dans la base test
from sklearn.model_selection import train_test_split
train, test = train_test_split(transactions_num,test_size=1000,random_state=1, stratify = transactions_num.CA_fort)
print(train.CA_fort.value_counts(normalize=True)) ; print(test.CA_fort.value_counts(normalize=True))

#construction du modèle
from sklearn.tree import DecisionTreeClassifier ; arbreFirst = DecisionTreeClassifier()
X = train[['Products','Products_Category']] ; y = train.CA_fort ; arbreFirst.fit(X, y)

#Evaluation du modele
eval_test = pd.DataFrame(arbreFirst.predict(test[['Products','Products_Category']]))

#matrice de confusion
from sklearn import metrics ; metrics.confusion_matrix(test.CA_fort,eval_test )

print(metrics.classification_report(test.CA_fort,eval_test)) #1-recall = tx faux + ou faux -

#taux de reconnaissance – (vrai positf + vrai négatif)/ effectif total (ici (849 + 62)/ 1000)
metrics.accuracy_score(test.CA_fort,eval_test )

#taux d'erreur – (faux positf + faux négatif)/ effectif total (ici (12 + 77)/ 1000)
1.0 - metrics.accuracy_score(test.CA_fort,eval_test )

#sensibilité est la capacité du modèle à prédire un positif quand la donnée est réellement positive
#Faux positif - sensibilité ici 62/(77+62) (2eme ligne)
metrics.recall_score(test.CA_fort,eval_test ,pos_label='oui')

#spécificité est la capacité du modèle à prédire un négatif lorsqu'il y a vraiment un négatif.
#Faux négatif - spécificité 1ere ligne)
metrics.recall_score(test.CA_fort,eval_test ,pos_label='non')

#précision – 62/(62+12) 2eme colonne
metrics.precision_score(test.CA_fort,eval_test,pos_label='oui')

#F1-score : moyenne harmonique entre rappel et précision :
metrics.f1_score(test.CA_fort,eval_test,pos_label='oui')
    
#arbre de decicision
from sklearn.tree import plot_tree
plot_tree(arbreFirst,feature_names = list(X.columns),filled=True)

#la variable qui influe le plus sur la nature du CA est le fait que la transaction dépasse les 13 produits 

#affichage plus grand pour une meilleure lisibilité
import matplotlib.pyplot as plt
plt.figure(figsize=(150,90))
plot_tree(arbreFirst,feature_names = list(X.columns),filled=True)
plt.show()

#importance des variables
impVarFirst = {"Variable":X.columns,"Importance":arbreFirst.feature_importances_}
print(pd.DataFrame(impVarFirst).sort_values(by="Importance",ascending=False))

############################################ Variable Cat ######################################################

col = list(transactions.columns)              
del col[0];del col[5];del col[5];del col[5]

transactions_cat = pd.DataFrame(np.c_[transactions.iloc[:,1:6],transactions.iloc[:,[9]]], 
                                columns = col, index = transactions['ID_Transaction']) 

#pour les variable catégorielles il faut effectuer un encodage one hot,
#chaque modalité devient une variable qui prend 1 si l'individu la possede 0 sinon
transactions_cat_bis = pd.get_dummies(transactions_cat[transactions_cat.columns[:-1]])

col = list(transactions_cat_bis.columns) ; col.append('CA_fort') 
transactions_cat = pd.DataFrame(np.c_[transactions_cat_bis.iloc[:,0:27],transactions.iloc[:,[9]]], 
                                columns = col, index = transactions['ID_Transaction']) 

#on découpe la base en 2 avec la meme proprtion de modalité 
#sur la variable à modeliser dans la base train que dans la base test
from sklearn.model_selection import train_test_split
train, test = train_test_split(transactions_cat,test_size=1000,random_state=1, stratify = transactions_cat.CA_fort)
print(train.CA_fort.value_counts(normalize=True)) ; print(test.CA_fort.value_counts(normalize=True))

#construction du modèle
from sklearn.tree import DecisionTreeClassifier ; arbreFirst = DecisionTreeClassifier()
X = train[train.columns[:-1]] ; y = train.CA_fort ; arbreFirst.fit(X, y)

#Evaluation du modele
eval_test = pd.DataFrame(arbreFirst.predict(test[train.columns[:-1]]))

#matrice de confusion
from sklearn import metrics ; metrics.confusion_matrix(test.CA_fort,eval_test )

print(metrics.classification_report(test.CA_fort,eval_test)) #1-recall = tx faux + ou faux -

#taux de reconnaissance – (vrai positf + vrai négatif)/ effectif total 
metrics.accuracy_score(test.CA_fort,eval_test )

#taux d'erreur – (faux positf + faux négatif)/ effectif total 
1.0 - metrics.accuracy_score(test.CA_fort,eval_test )

#sensibilité est la capacité du modèle à prédire un positif quand la donnée est réellement positive
#Faux positif - sensibilité (2eme ligne)
metrics.recall_score(test.CA_fort,eval_test ,pos_label='oui')

#spécificité est la capacité du modèle à prédire un négatif lorsqu'il y a vraiment un négatif.
#Faux négatif - spécificité 1ere ligne)
metrics.recall_score(test.CA_fort,eval_test ,pos_label='non')

#précision – 2eme colonne
metrics.precision_score(test.CA_fort,eval_test,pos_label='oui')

#F1-score : moyenne harmonique entre rappel et précision :
metrics.f1_score(test.CA_fort,eval_test,pos_label='oui')
    
#arbre de decicision
from sklearn.tree import plot_tree
plot_tree(arbreFirst,feature_names = list(X.columns),filled=True)

#affichage plus grand pour une meilleure lisibilité
import matplotlib.pyplot as plt
plt.figure(figsize=(150,90))
plot_tree(arbreFirst,feature_names = list(X.columns),filled=True)
plt.show()

#importance des variables
impVarFirst = {"Variable":X.columns,"Importance":arbreFirst.feature_importances_}
print(pd.DataFrame(impVarFirst).sort_values(by="Importance",ascending=False))

########################################### Variable Cat & Num ######################################################

#On y ajoute les variables numériques
col = list(transactions_cat_bis.columns)              
col.append('Products');col.append('Products_Category');col.append('CA_fort');
transactions_class = pd.DataFrame(np.c_[transactions_cat_bis.iloc[:,0:27], transactions.iloc[:,6:8],
                                transactions.iloc[:,[9]]], columns = col, index = transactions['ID_Transaction'])

#on découpe la base en 2 avec la meme proprtion de modalité 
#sur la variable à modeliser dans la base train que dans la base test
from sklearn.model_selection import train_test_split
train, test = train_test_split(transactions_class,test_size=1000,random_state=1, stratify = transactions_class.CA_fort)
print(train.CA_fort.value_counts(normalize=True)) ; print(test.CA_fort.value_counts(normalize=True))

#construction du modèle
from sklearn.tree import DecisionTreeClassifier ; arbreFirst = DecisionTreeClassifier()
X = train[train.columns[:-1]] ; y = train.CA_fort ; arbreFirst.fit(X, y)

#Evaluation du modele
eval_test = pd.DataFrame(arbreFirst.predict(test[train.columns[:-1]]))

#matrice de confusion
from sklearn import metrics ; metrics.confusion_matrix(test.CA_fort,eval_test )

print(metrics.classification_report(test.CA_fort,eval_test)) #1-recall = tx faux + ou faux -

#taux de reconnaissance – (vrai positf + vrai négatif)/ effectif total 
metrics.accuracy_score(test.CA_fort,eval_test )

#taux d'erreur – (faux positf + faux négatif)/ effectif total
1.0 - metrics.accuracy_score(test.CA_fort,eval_test )

#sensibilité est la capacité du modèle à prédire un positif quand la donnée est réellement positive
#Faux positif - sensibilité (2eme ligne)
metrics.recall_score(test.CA_fort,eval_test ,pos_label='oui')

#spécificité est la capacité du modèle à prédire un négatif lorsqu'il y a vraiment un négatif.
#Faux négatif - spécificité 1ere ligne)
metrics.recall_score(test.CA_fort,eval_test ,pos_label='non')

#précision –  2eme colonne
metrics.precision_score(test.CA_fort,eval_test,pos_label='oui')

#F1-score : moyenne harmonique entre rappel et précision :
metrics.f1_score(test.CA_fort,eval_test,pos_label='oui')
    
#arbre de decicision
from sklearn.tree import plot_tree
plot_tree(arbreFirst,feature_names = list(X.columns),filled=True)

#affichage plus grand pour une meilleure lisibilité
import matplotlib.pyplot as plt
plt.figure(figsize=(600,380))
plot_tree(arbreFirst,feature_names = list(X.columns),filled=True)
plt.show()

#importance des variables
impVarFirst = {"Variable":X.columns,"Importance":arbreFirst.feature_importances_}
print(pd.DataFrame(impVarFirst).sort_values(by="Importance",ascending=False))

#predire la natude des futures transaction
future_transac = pd.read_csv('future_transactions.csv', sep=",")
transactions_predit = pd.DataFrame(arbreFirst.predict(future_transac[future_transac.columns()])) 


