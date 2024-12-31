import time
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import requests
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import streamlit.components.v1 as components
import plotly.graph_objects as go
import io
import base64

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

@st.cache_resource(hash_funcs={plt.figure: lambda _: None})
def comparison_chart(df, id_client, feature):
    fig, ax = plt.subplots()
    sns.histplot(df, x=feature, color='black')
    mean = df[feature].mean()
    value_client = df.loc[df['SK_ID_CURR']==id_client, feature].iloc[0]

    ax.axvline(int(value_client), color="red", linestyle='--',linewidth=2, label =f'Client N° {int(id_client)}: {value_client}')
    ax.axvline(int(mean), color="orange", linestyle='--',linewidth=2, label ='Moyenne : %d'%mean)
    ax.set(title='Distribution %s' % feature, ylabel='')
    plt.legend()

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue())
    image = base64.b64decode(image_base64)
    return st.image(image)

#-------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title='Scoring Client',
    page_icon = "images/logo.jpg",
    initial_sidebar_state="expanded",
    layout="wide"
)

st.title("Scoring Client 🎖️")
st.sidebar.image('images/logo.jpg', use_container_width=True)

df, df_scaled_with_id, df_application_test = read_and_scale_data()

if not df.empty :
    with st.expander("Aperçu Données"):
        st.dataframe(df.head(5))
        st.dataframe(df_scaled_with_id.head(5))
        st.dataframe(df_application_test.head(5))

    # Selection ID client
    liste_clients = list(df['SK_ID_CURR'])
    col1, col2, col3 = st.columns(3)
    with col1:
        id_client = st.selectbox("Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant :",
                                (liste_clients))
        st.write(f"Vous avez sélectionné l'identifiant N° : **{int(id_client)}**")

    # Fiche Client
    with st.sidebar:
        st.write(f"**FICHE CLIENT N° {int(id_client)}** 📑")
        infos_client = df_application_test.loc[df_application_test['SK_ID_CURR']==id_client, ]
        st.write(" ")

        st.write("**Informations Client ============================**")
        st.write("**Sexe :**", infos_client['CODE_GENDER'].iloc[0])
        st.write("**Age :**", round(abs(infos_client['DAYS_BIRTH'].iloc[0]/365)))
        st.write('**Statut :**', infos_client['NAME_FAMILY_STATUS'].iloc[0])
        st.write("**Nombre d'enfants :**", infos_client['CNT_CHILDREN'].iloc[0])
        st.write("**Revenu total :**", infos_client['AMT_INCOME_TOTAL'].iloc[0],"$")
        st.write("**Niveau d'éducation :**", infos_client['NAME_EDUCATION_TYPE'].iloc[0])
        st.write("**Occupation :**", infos_client['OCCUPATION_TYPE'].iloc[0])

        st.write("**Informations Crédit ============================**")
        st.write("**Type Contrat :**", infos_client['NAME_CONTRACT_TYPE'].iloc[0])
        st.write("**Montant du crédit :**", infos_client['AMT_CREDIT'].iloc[0])
        st.write("**Montant du crédit annuel :**", infos_client['AMT_ANNUITY'].iloc[0])
        st.write("**Ratio crédit sur revenu :**", infos_client['AMT_INCOME_TOTAL'].iloc[0]/infos_client['AMT_CREDIT'].iloc[0])

        st.dataframe(infos_client)

    #===========================#
    # Prédiction Scoring Client #
    #===========================#
    feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    df_scaled_filtered = df_scaled_with_id.loc[df_scaled_with_id['SK_ID_CURR']==id_client, feats]

    predict_button = st.button('Prédire')
    if predict_button:
        prediction_df, proba_df = request_prediction(API_URL, data = df_scaled_filtered)

        with st.container(border=True):
            proba = round(proba_df["proba_classe_1"][0]*100, 2)
            prediction = round(prediction_df["y_pred"][0])
            prediction_df["decision"] = np.where(prediction_df.y_pred ==1, "Refusé", "Accordé")
            st.write(f"Le client **N° {int(id_client)}** a une probabilité de défaut de paiement estimé à : **{proba}%**")
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

    #====================#
    # Comparaison Client #
    #====================#
    st.divider()
    st.header("Comparaison Client avec les autres clients")
    feature_name = st.selectbox('Sélectionner un paramètre :', [
        "DAYS_BIRTH",
        "AMT_CREDIT",
        "DAYS_EMPLOYED",
        "AMT_INCOME_TOTAL",
        ])
    
    comparison_chart(df=df, id_client=id_client, feature=feature_name)

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
    st.subheader(f"Explication Locale Client N°{int(id_client)}")
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
