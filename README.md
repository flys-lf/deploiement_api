# deploiement_api
Projet réalisé dans le cadre d'un parcours certifiant de Data Scientist OpenClassroom : Implémentez un modèle de scoring (2 mois / 80h).

Repository contenant **le code de l'API de prédiction ainsi que le dashboard Streamlit appelant l'API** pour prédire la probabilité de défaut de paiement d'un client.

Les données en entrée sont générées avec le notebook ```LY_France_01_notebook_preprocessing_modelisation_112024.ipynb``` présent dans le repository contenant la modélisation : https://github.com/flys-lf/scoring_model/tree/main.

# Guideline
- Installer les dépendances :
```
poetry install
poetry shell
```

- Données disponibles ici : https://www.kaggle.com/c/home-credit-default-risk/data
- Lancer le script de ```preprocessing.py``` pour générer les données nettoyées nécessaires pour lancer le streamlit (CSV df_test_clean.csv et df_train_clean.csv)

- Lancer le Streamlit en local
```
streamlit run .\LY_France_Dashboard_112024.py
```

API de prédiction disponible à cet URL : https://scoringapi-ewckf3cxfrdbadhw.northeurope-01.azurewebsites.net