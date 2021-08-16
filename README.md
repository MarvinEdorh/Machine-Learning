# Linear Regression
Demo : https://github.com/MarvinEdorh/Machine-Learning/blob/main/Linear%20Regression.py

La régression linéaire est un modèle qui permet de modéliser toute variable numérique une combinainson linéaire d'autres variables numériques et d'un résidus aléatoire. En marketing digital ce modèle peut par exemple être utliser pour prédire la lifetime value d'un consommateur en fonction de sa durée de vie, de sa fréquence d'achat, de son panier moyen...Si l'on veut inclure au modele des variables catégorielles comme le device, les categories produits visitées, alors on peut soit effectuer un label encoding de ces variables ou alors réaliser une analyse factorielle afin de réduire le residus.

# Classification & Decisions Tree
Demo : https://github.com/MarvinEdorh/Machine-Learning/blob/main/Classification%20%26%20Decision%20Tree.py

Le modèle de classification permet de modéliser une variable binaire en fonction de varibles numériques. En marketing digital ce modèle peut être utiliser pour prédire le churn (l'abonné est sur le point de se désabonner ou non) ou la consommation (le client est sur le point de consommer ou pas). Si l'on veut inclure des variables catégorielle au modèle il faut préalablement y appliquer un label encoding (chaque modalité de la variable catégorielle devient une variable numérique égale à 1 si l'individu la possede et 0 sinon). On divise ensuite le jeu de donnée en 2, l'un qui permettra au modèle de s'entrainer à modeliser la variable dépendante et qui lui permettra de se tester. L'évaluation du modele se fait à l'aide de la matrice de confusion qui donne le taux de faux positifs et de faux negatifs de la phase de test
![decisions tree](https://user-images.githubusercontent.com/83826055/129446293-262d9803-549c-486b-9fe8-426f16bf5a66.png)

# K-Means Clustering
Demo : https://github.com/MarvinEdorh/Machine-Learning/blob/main/Clustering%20K-Means.py
