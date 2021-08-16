# Linear Regression
Demo : https://github.com/MarvinEdorh/Machine-Learning/blob/main/Linear%20Regression.py

La régression linéaire est un modèle qui permet de modéliser toute variable numérique une combinainson linéaire d'autres variables numériques et d'un résidus aléatoire. En marketing digital ce modèle peut par exemple être utliser pour prédire la lifetime value d'un consommateur en fonction de sa durée de vie, de sa fréquence d'achat, de son panier moyen...Si l'on veut inclure au modele des variables catégorielles comme le device, les categories produits visitées, alors on peut soit effectuer un label encoding de ces variables ou alors réaliser une analyse factorielle afin de réduire le residus.

# Classification & Decisions Tree
Demo : https://github.com/MarvinEdorh/Machine-Learning/blob/main/Classification%20%26%20Decision%20Tree.py

Le modèle de classification permet de modéliser une variable binaire (variable dépendante) en fonction de varibles numériques. En marketing digital ce modèle peut par exemple être utiliser pour prédire le churn (l'abonné est sur le point de se désabonner ou non) ou bien encore la consommation (le client est sur le point de consommer ou pas). Si l'on veut inclure des variables catégorielle au modèle il faut préalablement y appliquer un label encoding (chaque modalité de la variable catégorielle devient une variable numérique égale à 1 si l'individu la possede et 0 sinon). On divise ensuite au hazard le jeu de donnée en 2, un jeu d'apprentissage et un jeu de test, avec la même prortion de chaque modalité pour la variable dépendante dans chaque jeu de données. Le modèle va ensuite s'entrainer à modéliser la modalité "1" de variable dépendante avec le jeu d'aprentissage et puis se tester avec le jeu de test. L'évaluation du modèle se fait à l'aide de la matrice de confusion qui donne le taux de faux positifs et de faux négatifs de la phase de test. Ensuite l'abre de décision indique quelle variable contribue le plus au fait qu'un individu ait la modalité "1" pour la variable dépendante.

![decisions tree](https://user-images.githubusercontent.com/83826055/129446293-262d9803-549c-486b-9fe8-426f16bf5a66.png)

On voit ici quelle variable contribue le plus fait q'un individu effectue une transaction. Les variables catégorielles étant la résultante d'un lablel encoding elles ont pour modalité 0 ou 1 donc quand l'abre indique <0,5 cela veut dire = 0 et = 1 sinon. On constacte ici que la variale qui pédétermine le plus qu'un individus effectue une transaction soit que xxx soit égale 0, sinon que soit égale à 0, sinon que le nombre de visites dépasse 9,5 et ainsi de suite. On peut maintenant appliquer ce modèle sur de nouveaux individus pour voir les quels sont sur le point de consommer.

# K-Means Clustering
Demo : https://github.com/MarvinEdorh/Machine-Learning/blob/main/Clustering%20K-Means.py
