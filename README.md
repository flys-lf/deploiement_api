# deploiement_api
Projet réalisé dans le cadre d'un parcours certifiant de Data Scientist OpenClassroom : Implémentez un modèle de scoring (2 mois / 80h).

Repository contenant **le code de l'API de prédiction ainsi que le dashboard Streamlit appelant l'API** pour prédire la probabilité de défaut de paiement d'un client.

Repository contenant les fichiers non nécessaires au déploiement de l'API (notebook, data drift, pdf ...) ici : https://github.com/flys-lf/scoring_model

# Contexte :
Une société financière, nommée "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.
Elle souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

# Mission :
Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.

Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.

Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

# Guideline
- Installer les dépendances :
```
poetry install
poetry shell
```
- Données utilisées à placer dans un dossier ```input/``` : https://www.kaggle.com/c/home-credit-default-risk/data
- Les données nettoyées sont générées avec le notebook ```LY_France_01_notebook_preprocessing_modelisation_112024.ipynb``` (CSV df_test_clean.csv et df_train_clean.csv)

- Lancer MLFlow : 
```
mlflow server --host 127.0.0.1 --port 8080
```

- Lancer le Streamlit en local
```
streamlit run .\LY_France_Dashboard_112024.py
```

API de prédiction disponible à cet URL : https://scoringapi-ewckf3cxfrdbadhw.northeurope-01.azurewebsites.net

- Pour lancer l'API en local
```
fastapi dev .\LY_France_API_112024.py
```
  
