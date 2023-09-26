Dans ce projet, nous voulions développer un modèle capable d'accorder des crédits à des clients de la banque avec une forte confiance. 

Plusieurs algorithmes de machine learning ont été testées.

Mais le modèle ainsi développé est un algorithme de régression logarithmique.

Pour son développement, 200 features ont été sélectionnées (issues de feature engineering) issues des informations des clients.

Le modèle a été optimisé pour réduire les erreurs de faux négatifs (clients prédits solvables mais en réalité insolvable).

Le dossier est classé de cette manière : 

/Notebooks : 
On y retrouve les notebooks de développement du modèle (du prétraitement au choix du modèle final)
L'analyse du datadrift.
Notebook sur le développement du dashboard

/app_api_model :
L'ensemble des fichiers nécessaires au déploiement du modèle sous la forme FAST-API et déployé sur HEROKU
main.py (FAST API) appel les fonctions de model.py

/app_dashboard 
Contient l'ensemble des fichiers nécessaire au déploiement d'une application dashboard écrit en DASH et déployé sur HEROKU
Contient un fichier test.csv déjà au norme pour tester le dashboard

/data_drift_rapport
Contient un fichier html contenant une analyse du datadrift

/mlruns_logs 
Contient le suivi mlflow des derniers développement des modèles

Note méthodologique 
Contient un rapport sur le projet, son développement et les résultats 