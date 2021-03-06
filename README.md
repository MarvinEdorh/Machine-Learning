# Linear Regression : Lifetime Value 
Demo : https://github.com/MarvinEdorh/Machine-Learning/blob/main/Linear%20Regression.py

La régression linéaire est un modèle qui permet de modéliser toute variable numérique comme une combinaison linéaire d'autres variables numériques et d'un résidu aléatoire. En marketing digital ce modèle peut par exemple être utilisé pour prédire la lifetime value d'un consommateur en fonction de son panier moyen, sa fréquence d'achat, sa durée de vie... Si l'on veut inclure au modèle des variables catégorielles comme le device ou les catégories produits visitées ou consommées, alors on peut soit effectuer un label encoding de ces variables ou alors réaliser une analyse factorielle afin de réduire le résidu.
# Classification : Transaction & Churn
Demo : https://github.com/MarvinEdorh/Machine-Learning/blob/main/Classification%20%26%20Decision%20Tree.py

Le modèle de classification permet de modéliser une variable binaire (variable dépendante) en fonction de variables numériques. En marketing digital ce modèle peut par exemple être utilisé pour prédire le churn (l'abonné est sur le point de se désabonner ou non) ou bien encore la consommation (le client est sur le point de consommer ou pas). Si l'on veut inclure des variables catégorielles au modèle il faut préalablement y appliquer un label encoding (chaque modalité de la variable catégorielle devient une variable numérique égale à 1 si l'individu la possède et 0 sinon). On divise ensuite au hasard le jeu de données en 2, un jeu d'apprentissage et un jeu de test, avec la même proportion de chaque modalité de la variable dépendante dans chaque jeu de données. Le modèle va ensuite s'entraîner à modéliser la première modalité de variable dépendante avec le jeu d'apprentissage et puis se tester avec le jeu de test. L'évaluation du modèle se fait à l'aide de la matrice de confusion qui donne le taux de faux positifs et de faux négatifs de la phase de test. L'arbre de décision montre ensuite quelles variables contribuent le plus au fait qu'un individu ait la première modalité pour la variable dépendante.

![decisions tree](https://user-images.githubusercontent.com/83826055/129543979-49f4f07a-b6d0-46c5-85ae-cebca9e7f984.png)

On voit ici quelles variables contribuent le plus au fait qu'un individu effectue une transaction. Les variables catégorielles étant la résultante d'un lablel encoding elles ont pour modalité 0 ou 1 donc quand l'arbre indique <0,5 cela veut dire = 0 et = 1 sinon. On constate ici que la variable qui prédétermine le plus qu'un individus effectue une transaction soit que product_category_Apparel soit égal à 0 (donc que l'individu ne l'ait pas visité), sinon que le nombre de visites soit inférieur à 3,5 et ainsi de suite. On applique enfin le modèle à de nouveaux individus pour voir lesquels sont sur le point d'effectuer une transaction.
# K-Means Clustering : Segmentation
Demo 1 : https://github.com/MarvinEdorh/Data-Mining/blob/main/README.md#factor-analyzes--clustering

Demo 2 : https://github.com/MarvinEdorh/Machine-Learning/blob/main/Clustering%20K-Means.py
