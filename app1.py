#from turtle import color, goto
import numpy as np
import dill as pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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
    df0 = pickle.load(open("Credit_global_dashb.pkl", "rb"))
    columns_to_keep = ['SK_ID_CURR','EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE_YEARS', 'BURO_CREDIT_ACTIVE_Active_MEAN',
                       'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
                       'YEARS_EMPLOYED',
                       'BURO_YEARS_CREDIT_MIN',
                       'BURO_YEARS_CREDIT_MEAN',
                       'BURO_YEARS_CREDIT_UPDATE_MEAN' ]
    
    
    df=df0[df0['TARGET'].isnull()]
    df = df[columns_to_keep]
    seuil = 0.87

    def predict_note_authentication(data):

        prediction = classifier.predict_proba(data)
        print(prediction)
        return prediction
    
    def disp_caract(feat1,feat2):
        dfok=df0[df0['TARGET']==0]
        dfok['ysize']=1
        dfok['Statut']='Accepté'
        dfnok=df0[df0['TARGET']==1]
        dfnok['ysize']=2
        dfnok['Statut']='Refusé'
        dfclient['ysize']=50
        dfclient['Statut']='Prospect'
        dfT=pd.concat([dfok,dfnok,dfclient])
        dfT["TARGET"] = dfT["TARGET"].astype(str) #convert to string
        fig = px.scatter(
        dfT,
        x=feat1,
        y=feat2,
        color="Statut",
        color_discrete_sequence=["green", "red", "blue"],
        size='ysize',symbol="Statut"
        )
    
        return fig
    
    @st.cache
    def profil_client(ID,frame_True):
  
        frame_True['AGE']=round(abs(frame_True['AGE_YEARS']/1)).astype(int)
        ID_c=int(ID)  
        #INFO CLIENT True
        info_client_t=frame_True[frame_True['SK_ID_CURR']==ID_c]
        enfant_c_t=info_client_t['CNT_CHILDREN'].item()
        age_c_t=info_client_t['AGE'].item()
        genre_c_t=info_client_t['CODE_GENDER'].item()
        region_c_t=info_client_t['REGION_RATING_CLIENT'].item()
        arr_cl=[]
        #arr_cl.append(ID_c)
        arr_cl.append(age_c_t)
        arr_cl.append(genre_c_t)
        arr_cl.append(enfant_c_t)
        arr_cl.append(region_c_t)
        frame_info_client=pd.DataFrame(arr_cl)
        frame_info_client=frame_info_client.T
        frame_info_client.columns=['Age','Genre','Nb_enfants','Code_Region']
        frame_info_client.index=[str(ID_c)]
        return frame_info_client.T
    
    @st.cache
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
    dfclient=df0[df0.SK_ID_CURR==numcli]
    dfclient['TARGET']=2
    list1=columns_to_keep
    list1.remove('SK_ID_CURR')

    st.sidebar.subheader('Profil client ')
    st.sidebar.write(profil_client(numcli,dfclient))
    
    
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
        fig = yfunctions.gauge_charts(proba,seuil)
        st.write(fig)
    with col15:
        # Empty column to center the elements
        st.write("")
    with col2:
        st.image('./img/Feat_import.png', width=500)

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
    #### Mono & Bi analysis
    ### Dist Plot
    col1, col2, col3 = st.columns((3))
    
    with col1:
        feature1 = st.selectbox('Choisissez la 1ère caractéristique:',list1)
        #valueCustomer1 = dataCustomer.loc[dataCustomer[loanColumn]==user_input, feature1].values[0]
        #fig = utils.plotDistOneFeature(dataRef, feature1, valueCustomer1)
        #st.write(fig)
        #list2=list1.remove(feature1)
        
    with col2:
        
        feature2 = st.selectbox('Choisissez la 2nd caractéristique:',list1)
        #valueCustomer2 = dataCustomer.loc[dataCustomer[loanColumn]==user_input, feature2].values[0]
        #fig = utils.plotDistOneFeature(dataRef, feature2, valueCustomer2)
        #st.write(fig)
    
    with col3:
        #list3=list2.remove(feature2)
        feature3 = st.selectbox('Choisissez la 3ème caractéristique:',list1)
    
    #### Scatter Plot
    col1, col2 = st.columns(2)
    
    with col1:
        #listValueCustomer = [[feature1,valueCustomer1],[feature2,valueCustomer2]]
        fig1 = disp_caract(feature1,feature2)
        st.markdown('### ↓ Positionnement du prospect en fonction des 2 premières caractéristiques selectionnées')
        st.write(fig1)
    
    with col2:
        st.markdown('### ↓ Positionnement du prospect en fonction des caractéristiques 1 et 3')
        fig2 = disp_caract(feature1,feature3)
        st.write(fig2)

if __name__=='__main__':
    main()
    
    
    
