import numpy as np
import dill as pickle
import pandas as pd
import codecs
import yfunctions
import streamlit.components.v1 as components
#import utils
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)


#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])

def display(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r',encoding='utf-8')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)







def main():

    st.set_page_config(
        # layout="centered",
        layout='wide',
        initial_sidebar_state="collapsed"
    )
    pickle_in = open("ylh_lgbm.pkl", "rb")
    classifier = pickle.load(pickle_in)

    explainer = pickle.load(open("lime_explainer.pickle", "rb"))
    df = pickle.load(open("Credit_global_test.pkl", "rb"))
    columns_to_keep = ['SK_ID_CURR','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE_YEARS', 'BURO_CREDIT_ACTIVE_Active_MEAN',
                       'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
                       'YEARS_EMPLOYED',
                       'BURO_YEARS_CREDIT_MIN',
                       'BURO_YEARS_CREDIT_MEAN',
                       'BURO_YEARS_CREDIT_UPDATE_MEAN' ]
    df = df[columns_to_keep]
    seuil = 0.5

    def predict_note_authentication(data):

        prediction = classifier.predict_proba(data)
        print(prediction)
        return prediction

    def lime_show(data):

        X_explain = data

        # Explaining first subject in test set using all 30 features
        exp = explainer.explain_instance(X_explain.values[0, :], classifier.predict_proba,
                                         num_features=30)
        # Plot local explanation
        plt = exp.as_pyplot_figure()
        plt.tight_layout()
        # xp.show_in_notebook(show_table=True)
        return exp

    numcli=st.sidebar.selectbox("Numéro de client : ", df['SK_ID_CURR'].tolist()	)

    numcli=int(numcli)
    ########### Top ##############################
    col1, col15, col2 = st.columns((2, 1, 3))
    with col1:
        st.image('./img/logo.png', width=300)
    with col15:
        # Empty column to center the elements
        st.write("")
    with col2:
        st.title('   Simulation de prêt')
        st.header('  Obtenez une réponse instantanément')


    dfchoice=df[df.SK_ID_CURR==numcli]
    X=dfchoice.drop('SK_ID_CURR',axis=1)
    'Sélection du prêt du client n° :  ', numcli
    st.write(dfchoice)

    #result=np.array([[0, 0]])
    proba=predict_note_authentication(X)[0][0]

    affichage= 'Le score est de : {}'.format(round(proba,3))

    if proba>=seuil:
        affichage += " , le dossier est accepté!"
        st.success(affichage)
    else :
        affichage += ", le dossier est refusé!"
        st.error(affichage)


    col1, col15, col2 = st.columns((6, 1, 8))
    with col1:
        fig = yfunctions.gauge_charts(proba)
        st.write(fig)
    with col15:
        # Empty column to center the elements
        st.write("")
    with col2:
        st.image('./img/Feat_import.jpg', width=500)

    st.write("## Détail des composantes du score pour le client ")
    exp=lime_show(X)
    exp.save_to_file("test.jpg")
    plt = exp.as_pyplot_figure()
    plt.savefig('lime_report.jpg')
    exp.save_to_file('lime_report.html')
    HtmlFile = open("lime_report.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    col1, col15, col2 = st.columns((2, 1, 3))
    with col1:
        st.image('lime_report.jpg')
    with col15:
        # Empty column to center the elements
        st.write("")
    with col2:
        display('lime_report.html')


if __name__=='__main__':
    main()
    
    
    
