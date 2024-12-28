import time
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
import plotly.graph_objects as go

# API_URL = "http://127.0.0.1:8000/predict"
API_URL = "https://scoringapi-ewckf3cxfrdbadhw.northeurope-01.azurewebsites.net/predict"

MODEL_FILE = 'model_LGBM.pkl'

def request_prediction(url, data):
    payload = data.to_json()
    response = requests.post(url, data= payload)

    # Check the HTTP response status code
    if response.status_code == 200:
        # Parse and print the JSON response (assuming it contains the prediction)
        result = response.json()
        prediction_df = pd.DataFrame.from_dict(result["prediction"])
        proba_df = pd.DataFrame.from_dict(result["probability"])
        print(prediction_df)
        print(proba_df)
    else:
        # Handle the case where the API request failed
        print(f'API Request Failed with Status Code: {response.status_code}')
        print(f'Response Content: {response.text}')
    return prediction_df, proba_df

@st.cache_resource(max_entries=1, ttl=3600 * 24)
def read_and_scale_data(num_rows = None):
    df_application_test = pd.read_csv('input/application_test.csv', nrows= num_rows)
    df = pd.read_csv('data/df_test_clean.csv', nrows= num_rows)

    with st.spinner('Reading In Progress...'):
        time.sleep(5)
        # Preprocessing input data
        feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

        # Scaling data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[feats])
        df_scaled = pd.DataFrame(scaled_data, columns=list(df[feats]))
        df_scaled_with_id = pd.concat([df_scaled, df['SK_ID_CURR']], axis=1)
    return df, df_scaled_with_id, df_application_test

@st.cache_resource(hash_funcs={plt.figure: lambda _: None})
def compute_shap_values():
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_shap)
    return explainer, shap_values

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

@st.cache_resource(hash_funcs={plt.figure: lambda _: None})
def plot_shap_values(values_shap, shap_df, type = None, sort = None):
    fig, ax = plt.subplots(figsize=(15,5))
    shap.summary_plot(values_shap, shap_df, plot_type=type, sort = sort)
    st.pyplot(fig)


#-------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title='Scoring Client',
    page_icon = "images/logo.jpg",
    initial_sidebar_state="expanded",
    layout="wide"
)

# st.image("images/banner.jpg")
st.title("Scoring Client 🎖️")

df, df_scaled_with_id, df_application_test = read_and_scale_data()

if not df.empty :
    with st.expander("Aperçu Données"):
        st.dataframe(df.head(5))
        st.dataframe(df_scaled_with_id.head(5))
        st.dataframe(df_application_test.head(5))

    #===========================#
    # Prédiction Scoring Client #
    #===========================#

    liste_clients = list(df['SK_ID_CURR'])
    col1, col2, col3 = st.columns(3)
    with col1:
        id_client = st.selectbox("Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant :",
                                (liste_clients))
        st.write(f"Vous avez sélectionné l'identifiant N° : **{id_client}**")


    with st.expander("Fiche Client 📑"):
        st.write(f"**Client N° {id_client}**")
        infos_client = df_application_test.loc[df_application_test['SK_ID_CURR']==id_client, ]
        col1, col2 = st.columns(2)
        with col1:
             st.write("Informations Client ------------")
             st.write("**Sexe :**", infos_client['CODE_GENDER'].iloc[0])
             st.write("**Age :**", round(abs(infos_client['DAYS_BIRTH'].iloc[0]/365)))
             st.write('**Statut :**', infos_client['NAME_FAMILY_STATUS'].iloc[0])
             st.write("**Nombre d'enfants :**", infos_client['CNT_CHILDREN'].iloc[0])
             st.write("**Revenu total :**", infos_client['AMT_INCOME_TOTAL'].iloc[0],"$")
             st.write("**Niveau d'éducation :**", infos_client['NAME_EDUCATION_TYPE'].iloc[0])
             st.write("**Occupation :**", infos_client['OCCUPATION_TYPE'].iloc[0])
             

        with col2:
            st.write("Informations Crédit ------------")
            st.write("**Type Contrat :**", infos_client['NAME_CONTRACT_TYPE'].iloc[0])
            st.write("**Montant du crédit :**", infos_client['AMT_CREDIT'].iloc[0])
            st.write("**Montant du crédit annuel :**", infos_client['AMT_ANNUITY'].iloc[0])
            st.write("**Ratio crédit sur revenu :**", infos_client['AMT_INCOME_TOTAL'].iloc[0]/infos_client['AMT_CREDIT'].iloc[0])

        
        st.dataframe(infos_client)

    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    df_scaled_filtered = df_scaled_with_id.loc[df_scaled_with_id['SK_ID_CURR']==id_client, feats]

    # Prédiction
    predict_button = st.button('Prédire')
    if predict_button:
        prediction_df, proba_df = request_prediction(API_URL, data = df_scaled_filtered)

        with st.container(border=True):
            proba = round(proba_df["proba_classe_1"][0]*100, 2)
            prediction = round(prediction_df["y_pred"][0])
            prediction_df["decision"] = np.where(prediction_df.y_pred ==1, "Refusé", "Accordé")
            st.write(f"Le client **N° {id_client}** a une probabilité de défaut de paiement estimé à : **{proba}%**")
            decision = prediction_df["decision"][0]
            if decision == "Accordé" :
                st.success(f"Crédit {decision}")
            elif decision == "Refusé" :
                st.error(f"Crédit {decision}")

        # Jauge score client ================================================
        fig = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = proba,
            mode = "gauge+number",
            title = {'text': "Score crédit du client", 'font': {'size': 24}},
            delta = {'reference': 50},
            gauge = {'axis': {'range': [None, 100],
                            'tickwidth': 3,
                            'tickcolor': 'black'},
                    'bar': {'color': 'white', 'thickness' : 0.15},
                    'bgcolor': 'white',
                    'borderwidth': 2,
                    'bordercolor': 'gray',
                    'steps': [{'range': [0, 20], 'color': '#D0E4D0'},
                            {'range': [20, 52], 'color': '#B1EABF'},
                            {'range': [52, 67], 'color': '#F7D78C'},
                            {'range': [67, 80], 'color': '#EF9B57'},
                            {'range': [80, 100], 'color': '#EF6257'}],
                    'threshold': {'line': {'color': 'white', 'width': 5},
                                'thickness': 0.20,
                                'value': proba}}))

        fig.update_layout(paper_bgcolor='white',
                                height=300, width=400,
                                font={'color': 'black', 'family': 'Calibri'},
                                margin=dict(l=0, r=0, b=0, t=0, pad=0))
        st.plotly_chart(fig, use_container_width=True)

        #=============================================================#
        # Comparaison du profil du client à son groupe d'appartenance #
        #=============================================================#
        # feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        # df_shap = df_scaled_with_id[feats]
        # explainer, shap_values = compute_shap_values()

        # df_scaled = df_scaled_with_id[feats]

        # # Lecture des probas pour tout le df
        # model = pickle.load(open(f"{MODEL_FILE}", "rb"))
        # prediction = (model.predict_proba(df_scaled)[:,1] >= 0.52).astype(float) # utilisation du seuil optimal
        # probability = model.predict_proba(df_scaled)
        # prediction_df = pd.DataFrame(prediction, columns=['y_pred'])
        # probability_df = pd.DataFrame(probability, columns=['proba_classe_0', 'proba_classe_1'])

        # proba = round(probability_df["proba_classe_1"][0]*100, 2)
        # prediction = round(prediction_df["y_pred"][0])
        # prediction_df["decision"] = np.where(prediction_df.y_pred ==1, "Refusé", "Accordé")

        

        # # Titre 1
        # st.markdown("""
        #             <br>
        #             <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
        #             2. Comparaison du profil du client à celui des clients dont la probabilité de défaut de paiement est proche</h1>
        #             """, 
        #             unsafe_allow_html=True)
        # st.write("Pour la définition des groupes de clients, faites défiler la page vers le bas.")

        # # Calcul des valeurs Shap
        # explainer_shap = shap.TreeExplainer(model)
        # st.write(shap_values[1])
        # # shap_values = explainer_shap.shap_values(df_shap)
        # shap_values_df = pd.DataFrame(data=shap_values, columns=df_shap.columns)
        # st.dataframe(shap_values_df.head(5))


        # df_groupes = pd.concat([probability_df['proba_classe_1'], shap_values_df], axis=1)
        # st.dataframe(df_groupes.head(5))

        # df_groupes['typologie_clients'] = pd.qcut(df_groupes.proba_classe_1,
        #                                             q=5,
        #                                             precision=1,
        #                                             labels=['20%_et_moins',
        #                                                     '21%_30%',
        #                                                     '31%_40%',
        #                                                     '41%_60%',
        #                                                     '61%_et_plus'])

        # # Titre H2
        # st.markdown("""
        #             <h2 style="color:#418b85;text-align:left;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
        #             Comparaison de “la trajectoire” prise par la prédiction du client à celles des groupes de Clients</h2>
        #             """, 
        #             unsafe_allow_html=True)
        # st.write("")

        # # Moyenne des variables par classe
        # df_groupes_mean = df_groupes.groupby(['typologie_clients']).mean()
        # df_groupes_mean = df_groupes_mean.rename_axis('typologie_clients').reset_index()

        # st.dataframe(df_groupes_mean)
        # df_groupes_mean["index"]=[1,2,3,4, 5]
        # df_groupes_mean.set_index('index', inplace = True)

        # # récupération de l'index correspondant à l'identifiant du client
        # # idx = int(lecture_X_test_clean()[lecture_X_test_clean()['sk_id_curr']==ID_client].index[0])
        # idx = int(df_scaled_with_id[df_scaled_with_id['SK_ID_CURR']==id_client].index[0])

        # # dataframe avec shap values du client et des 5 groupes de clients
        # comparaison_client_groupe = pd.concat([df_groupes[df_groupes.index == idx],
        #                                         df_groupes_mean],
        #                                         axis = 0)
        # comparaison_client_groupe['typologie_clients'] = np.where(comparaison_client_groupe.index == idx,
        #                                                         df_scaled_with_id.iloc[idx, 0],
        #                                                         comparaison_client_groupe['typologie_clients'])
        # # transformation en array
        # nmp = comparaison_client_groupe.drop(
        #                     labels=['typologie_clients', "proba_classe_1"], axis=1).to_numpy()

        # fig = plt.figure(figsize=(8, 20))
        # shap.decision_plot(explainer_shap.expected_value[0], 
        #                             nmp, 
        #                             feature_names=comparaison_client_groupe.drop(
        #                                             labels=['typologie_clients', "proba_classe_1"], axis=1).columns.to_list(),
        #                             feature_order='importance',
        #                             highlight=0,
        #                             legend_labels=['Client', '20%_et_moins', '21%_30%', '31%_40%', '41%_60%', '61%_et_plus'],
        #                             plot_color='inferno_r',
        #                             legend_location='center right',
        #                             feature_display_range=slice(None, -57, -1),
        #                             link='logit')
        # # st_shap(shap.decision_plot(explainer_shap.expected_value[0], 
        # #                             nmp, 
        # #                             feature_names=comparaison_client_groupe.drop(
        # #                                             labels=['typologie_clients', "proba_classe_1"], axis=1).columns.to_list(),
        # #                             feature_order='importance',
        # #                             highlight=0,
        # #                             legend_labels=['Client', '20%_et_moins', '21%_30%', '31%_40%', '41%_60%', '61%_et_plus'],
        # #                             plot_color='inferno_r',
        # #                             legend_location='center right',
        # #                             feature_display_range=slice(None, -57, -1),
        # #                             link='logit'))

        # # Titre H2
        # st.markdown("""
        #             <h2 style="color:#418b85;text-align:left;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
        #             Constitution de groupes de clients selon leur probabilité de défaut de paiement</h2>
        #             """, 
        #             unsafe_allow_html=True)
        # st.write("")

        # col1, col2 = st.columns(2)
        # with col1:
        #     fig1, ax1 = plt.subplots(figsize=(8, 6))
        #     plot_countplot(df=df_groupes, 
        #                 col='typologie_clients', 
        #                 order=False,
        #                 palette='rocket_r', ax=ax1, orient='v', size_labels=12)
        #     plt.title("Regroupement des Clients selon leur Probabilité de Défaut de Paiement\n",
        #             loc="center", fontsize=16, fontstyle='italic', fontname='Roboto Condensed')
        #     fig1.tight_layout()
        #     st.pyplot(fig1)
        # with col2:
        #     fig2, ax2 = plt.subplots(figsize=(8, 6))
        #     plot_aggregation(df=df_groupes,
        #                 group_col='typologie_clients',
        #                 value_col='proba_classe_1',
        #                 aggreg='mean',
        #                 palette="rocket_r", ax=ax2, orient='v', size_labels=12)
        #     plt.title("Probabilité Moyenne de Défaut de Paiement par Groupe de Clients\n",
        #             loc="center", fontsize=16, fontstyle='italic', fontname='Roboto Condensed')
        #     fig2.tight_layout()
        #     st.pyplot(fig2)

    #======================#
    # Analyse des Features #
    #======================#
    st.divider()
    st.header("Analyse des Features")

    model = pickle.load(open(f"{MODEL_FILE}", "rb"))
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    df_shap = df_scaled_with_id[feats]
    # explainer, shap_values, fig_barplot, fig_shap_plot = compute_shap_values()
    explainer, shap_values = compute_shap_values()

    # Explication Locale --------------------------------------------------------------------------------------------------------------
    st.subheader(f"Explication Locale Client N°{id_client}")
    # récupération de l'index correspondant à l'identifiant du client
    idx_selected = int(df_scaled_with_id[df_scaled_with_id['SK_ID_CURR']==id_client].index[0])
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    st_shap(
        shap.force_plot(
            explainer.expected_value,
            shap_values[idx_selected,:],
            df_shap.iloc[idx_selected,:],
            # df[feats].iloc[idx_selected,:],
            link='logit',
            ordering_keys=True,
            text_rotation=0,
            contribution_threshold=0.05
        )
    )

    col1, col2 = st.columns(2)
    with col1 :
        st.subheader("Explication Globale")
        plot_shap_values(shap_values, df_shap, type='bar', sort = True)

    with col2:
        st.subheader("Explication Locale")
        plot_shap_values(shap_values, df_shap, sort = True)
