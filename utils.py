# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:44:00 2021

@author: Jérémie Aucher
"""
### Declaration ###
import pickle
import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
import base64
import shap
import plotly.graph_objects as go
import plotly.express as px


# Initialisation ###
loanColumn = 'SK_ID_CURR'
target = 'TARGET'
colorLabel='Demande de prêt:'
colW=350
colH=500
url = "https://ja-p7-api.herokuapp.com/"
minRating=-1
maxRating=1
localThreshold=0


### For API Asking ###
def convToB64(data):
    '''
    As input: <data> of any kind.
    The function converts the <data> to base-64 then the resulting string is encoded in UTF-8.
    Output: The result obtained.
    '''
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')

def restoreFromB64Str(data_b64_str):
    '''
    Input: Data converted to Base-64 and then encoded to UTF-8. 
          Ideally data from the convToB64 function.
    The function restores the encoded data to its original format.
    Output: The restored data
    '''
    return pickle.loads(base64.b64decode(data_b64_str.encode()))

def askAPI(apiName, url=url, params=None):
    '''
    This function allows to query an API.
    It has been designed to query an API running on a FLASK server.
    The data received from the API must be in base-64 format encoded in UFT-8.
    Input:
        <apiName>: Name of the API to query, this is the name found at the end 
                   of the URL before the parameter list.
        <url>: url where the API to query is stored.
        <params>: At None by default if no parameters to send. 
                  The parameters sent must be in dictionary format {'Parameter Name': 'Parameter Value'}
    Output: The response provided by the API, decoded through the restoreFromB64Str function.
    '''
    url=url+str(apiName)
    resp = requests.post(url=url,params=params).text
    return restoreFromB64Str(resp)


def splitAndAskAPI(data):
    '''
    To get around a limitation of the Heroku server when using the free package 
    it may be necessary to split the data to be sent to the API so that 
    it can be reconstituted on the server side.
    To be used, the remote API must be designed to handle this kind of data.
    '''
	# Split and send data to the API
    print('splitAndAskAPI')
    print(askAPI(apiName='initSplit'))
    if askAPI(apiName='initSplit'):
        print('OK')
        for j,i in enumerate(splitString(data, 5)):
	#         time.sleep(1)
	        print(f'len du split n°{j}: {len(i)}')
	        if askAPI(apiName='merge', params=dict(txtSplit=i, numSplit=j)):
	            print('OK... pour le moment...')
	        else:
	            print('Aie Aie Aie...')
        resp = askAPI(apiName='endSplit')
        print(resp)
        return resp

@st.cache(suppress_st_warning=True)
def apiModelPrediction(data,loanNumber,columnName='SK_ID_CURR',url=url):
    '''
    This function allows you to make a prediction by querying the remote model via an API.
    The function supports Heroku's limitation on the size of data that can be sent in a POST request.
    As input: 
        <data> the data in pandas dataframe format compatible with the model being queried.
        <loanNumber> the loan number of the client to be queried
        <columnName> The name of the column containing the loan numbers
        <url> URL of the API to query'
    In output: The prediction of the model according to the format (Exact prediction (0 or 1), probabilistic prediction [0;1])
    '''
    print('apiModelPrediction')
    # Preparation of the information to be sent
    # Recovering the index
    idx = getTheIDX(data,loanNumber,columnName)
    print(f'idx={idx}')
	# Creation of a Pandas Series containing the customer's information
    dataOneCustomer = data.iloc[[idx]].values
	# Data encoding in base64 then in String UTF-8 format
    dataOneCustomerB64Txt = convToB64(dataOneCustomer)
	# The data is sent in 5 parts to bypass a data volume limitation on Heroku
    dictResp = splitAndAskAPI(data=dataOneCustomerB64Txt)
    
    return dictResp['predExact'], dictResp['predProba']

### Load Data and More ###
@st.cache(suppress_st_warning=True)
def loadData():
    '''
    This function returns the data in pickle format that are contained in the "pickle" folder
    The data returned are:
        'dataRef.pkl' which contains the data of the customer base which was used to train 
        the model and which will be used for the realization of the various graphics of the dashboard.
        dataCustomer' which contains a list of customers that can be queried to know 
        if their loan request is granted or not.
    '''
    return pickle.load(open(os.getcwd()+'/pickle/dataRef.pkl', 'rb')),\
        pickle.load(open(os.getcwd()+'/pickle/dataCustomer.pkl', 'rb'))

@st.cache(suppress_st_warning=True)
def loadModel(modelName='model'):
    '''
    This function queries and returns the model stored on the remote server.
    '''
    return askAPI(apiName=modelName)

@st.cache(suppress_st_warning=True)
def loadThreshold():
    '''
    This function queries and returns the value 
    of the threshold (from the LGBMClassifier model) 
    stored on the remote server.
    '''
    return askAPI(apiName='threshold')

@st.cache(suppress_st_warning=True)
def loadRatingSystem():
    '''
    This function queries and returns the value 
    of the rating system (from the LGBMClassifier model) 
    stored on the remote server.
    As Input: Nothing
    As output: minimum score, maximum score, theshold
    '''
    return askAPI(apiName='ratingSystem')

### Get Data ###
@st.cache(suppress_st_warning=True)
def getDFLocalFeaturesImportance(model,X,loanNumber,nbFeatures=12,inv=False):
    '''
    This function returns the Pandas dataframe which is used 
    to make the graph of features importance with the SHAP library.
    This allows to realize a graph using another graphical library 
    than the one used by default with SHAP.
    '''
    idx = getTheIDX(data=X,columnName=loanColumn,value=loanNumber)
    shap_values = shap.TreeExplainer(model).shap_values(X.iloc[[idx]])[0]
    
    if inv:
        shap_values *= -1
    
    dfShap = pd.DataFrame(shap_values, columns=X.columns.values)
    serieSignPositive = dfShap.iloc[0,:].apply(lambda col: True if col>=0 else False)

    serieValues = dfShap.iloc[0,:]
    serieAbsValues = abs(serieValues)
    return pd.DataFrame(
        {
            'values':serieValues,
            'absValues':serieAbsValues,
            'positive':serieSignPositive,
            'color':map(lambda x: 'red' if x else 'blue', serieSignPositive)
            }
        ).sort_values(
            by='absValues',
            ascending=False
            ).iloc[:nbFeatures,:].drop('absValues', axis=1)

def getTheIDX(data,value,columnName='SK_ID_CURR'):
    '''
    Returns the index corresponding to the 1st value contained 
    in value contained in the column columnName of the Dataframe data.
    ''' 
    return data[data[columnName] == value].index[0]

def splitString(t, nbSplit):
    '''
    This function splits a string into <nbSplit> pieces.
    If possible, all pieces of text have the same number of characters.
    Returns the split string in List format.
    '''
    import textwrap
    from math import ceil
    return textwrap.wrap(t, ceil(len(t)/nbSplit))

@st.cache(suppress_st_warning=True)
def get_df_global_shap_importance(model, X):
    '''
    This function returns the Pandas dataframe used to create the global features importance graph.
    This allows to recreate the graph initially provided by SHAP with another graphical library.
    '''
    # Explain model predictions using shap library:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]
    return pd.DataFrame(
        zip(
            X.columns[np.argsort(np.abs(shap_values).mean(0))][::-1],
            np.sort(np.abs(shap_values).mean(0))[::-1]
        ),
        columns=['feature','importance']
    )

@st.cache(suppress_st_warning=True)
def convertUpperAndLowerBoundAndThreshoold(value,
                                           oldMin,oldMax,oldThreshold,
                                           newMin,newMax,newThreshold):
    '''
    Convert a rating with a defined low and high bound into another rating 
    that can have different low and high bounds.
    It is possible to define thresholds that define the limit rating of 
    what could be NOK/OK before and after.
    For example: 
    A model returns a score between 0 and 1 with a threshold at 0.8.
    Below 0.8 the result is NOK, above 0.8 the result is OK.
    We want to display a score between -1 and 1 with a threshold of 0, 
    independently of the threshold defined before for readability concerns 
    if the score would be for example presented to a customer who would 
    not know this notion of threshold and would prefer, intuitively imagine 
    a strictly negative score as NOK and a positive score as OK.
    As input: 
    - value: the score to convert
    - oldMin: low limit in the scoring system of the score to convert
    - oldMax: high limit in the scoring system of the score to convert
    - oldThreshold: threshold in the rating system of the score to convert
    - newMin: low limit in the rating system of the rating to be converted
    - newMax: high limit in the rating system of the converted score
    - newThreshold: threshold in the rating system of the converted score
    As output:
    - the score in the new rating system
    '''
    
    if value < oldThreshold:
        oldMax=oldThreshold
        newMax=newThreshold
    else:
        oldMin=oldThreshold
        newMin=newThreshold
        
  
    return ((value-oldMin)*((newMax-newMin)/(oldMax-oldMin)))+newMin

### Plot Chart ###
@st.cache(suppress_st_warning=True)
def gauge_chart(score, minScore, maxScore, threshold):
    '''
    Returns a gauge figure with a number of predefined parameters.
    All you have to do is to give it the <score> as well as the <threshold>.
    '''
    
    convertedScore = convertUpperAndLowerBoundAndThreshoold(value=score,
                                                            oldMin=minScore,
                                                            oldMax=maxScore,
                                                            oldThreshold=threshold,
                                                            newMin=minRating,
                                                            newMax=maxRating,
                                                            newThreshold=localThreshold)
    
    
    color="RebeccaPurple"
    if convertedScore<localThreshold:
        color="darkred"
    else:
        color="green"
    fig = go.Figure(
        go.Indicator(
            domain = {
                'x': [0, 0.9],
                'y': [0, 0.9]
                },
            value = convertedScore,
            mode = "gauge+number+delta",
            title = {
                'text': "Score"
                },
            gauge = {
                'axis':{
                    'range':[-1, 1]
                    },
                'bar': {
                    'color': color
                    },
                'steps' : [
                 {
                     'range': [-1, -0.8],
                     'color': "#ff0000"
                     },
                 {
                     'range': [-0.8, -0.6],
                     'color': "#ff4d00"
                     },
                 {
                     'range': [-0.6, -0.4],
                     'color': "#ff7400"
                     },
                 {
                     'range': [-0.4, -0.2],
                     'color': "#ff9a00"
                     },
                 {
                     'range': [-0.2, 0],
                     'color': "#ffc100"
                     },
                 {
                     'range': [0, 0.2],
                     'color': "#c5ff89"
                     },                 
                 {
                     'range': [0.2, 0.4],
                     'color': "#b4ff66"
                     },
                 {
                     'range': [0.4, 0.6],
                     'color': "#a3ff42"
                     },
                 {
                     'range': [0.6, 0.8],
                     'color': "#91ff1e"
                     },
                 {
                     'range': [0.8, 1],
                     'color': "#80f900"
                     }
                 ],
             'threshold' :{
                 'line':{
                     'color': color,
                     'width': 8
                     },
                 'thickness': 0.75,
                 'value': convertedScore
                 }
             },
            delta = {'reference': 0.5, 'increasing': {'color': "RebeccaPurple"}}
            ))
    return fig

@st.cache(suppress_st_warning=True)
def plotGlobalFeaturesImportance(model, X, nbFeatures=10):
    '''
    Returns a figure allowing the display of the global features importance.
    The calculation is done by the SHAP library and the display of the graph with plotly.
    '''
    # Removal of the <target> column if it exists.
    X = X.drop(target, axis=1, errors='ignore')
    
    data = get_df_global_shap_importance(model, X)
    y=data.head(nbFeatures)['importance']
    x=data.head(nbFeatures)['feature']
    
    fig = go.Figure(
        data=[go.Bar(x=x, y=y,marker=dict(color=y, colorscale='viridis'))],
        layout=go.Layout(
        title=go.layout.Title(text="Importance Globale des Caractéristiques:")
        )
    )

    return fig

@st.cache(suppress_st_warning=True)
def plotLocalFeaturesImportance(model,X,loanNumber,nbFeatures=12):
    '''
    Returns a figure allowing the display of the globallocal features importance.
    The calculation is done by the SHAP library and the display of the graph with plotly.
    '''
    dfValuesSign = getDFLocalFeaturesImportance(
        model=model,
        X=X,
        loanNumber=loanNumber,
        nbFeatures=nbFeatures,
        inv=False
        )
    i = dfValuesSign.index
    fig = px.bar(dfValuesSign,
                 x='values',
                 y=i,
                 color='color',
                 orientation='h',
                 category_orders=dict(index=list(i)))
    fig.update_layout(
        title="Importance des Caractéristiques du Client:",
        yaxis={'title': None},
        xaxis={'title': None},
        showlegend=False
        )
    return fig

def adaptTargetValuesAndTitle(data):
    data = data.copy()
    data[target] = data[target].map({0:'Accepté', 1:'Refusé'})
    return data.rename(columns={target:'Demande de prêt:'})

@st.cache(suppress_st_warning=True)
def plotDistOneFeature(dataRef,feature,valCust):
    dataRef = adaptTargetValuesAndTitle(dataRef)
    fig = px.histogram(dataRef,
                       x=feature,
                       color=colorLabel,
                       marginal="box",
                       histnorm='probability')  # can be `box`, `violin`
    fig.add_vline(x=valCust, line_width=3, line_dash="dash", line_color="red")
    return fig

@st.cache(suppress_st_warning=True)
def plotScatter2D(dataRef, listValCust):
    '''
    Returns a figure generated by plotly express.
    The figure is a scatter plot in 2 dimensions representing all the customers.
    Also will be displayed two red lines (one vertical and the other horizontal' ) 
    whose intersection represents the location of the observed customer.
    '''
    dataRef = adaptTargetValuesAndTitle(dataRef)
    
    fig = px.scatter(
        dataRef,
        x=listValCust[0][0],
        y=listValCust[1][0],
        color=colorLabel,
        color_discrete_sequence=px.colors.qualitative.Plotly
        )
    
    fig.add_vline(x=listValCust[0][1], line_width=1, line_dash="solid", line_color="red")
    fig.add_hline(y=listValCust[1][1], line_width=1, line_dash="solid", line_color="red")
    
    fig.update_layout(showlegend=True)
    
    return fig

@st.cache(suppress_st_warning=True)
def plotScatter3D(dataRef, listValCust):
    '''
    Returns a figure generated by plotly express.
    The figure is a scatter plot in 3 dimensions representing all the customers. 
    The color of the points is realized according to the feature <feature>
    The observed customer is represented by a point of a different color than all the others.
    '''
    
    dataRef = adaptTargetValuesAndTitle(dataRef)
    
    fig = px.scatter_3d(
        dataRef,
        x=listValCust[0][0],
        y=listValCust[1][0],
        z=listValCust[2][0],
        color=colorLabel,
        color_discrete_sequence=px.colors.qualitative.Plotly
        )
    fig.update_layout(showlegend=True)
    
    fig.add_scatter3d(
        x=[listValCust[0][1]],
        y=[listValCust[1][1]],
        z=[listValCust[2][1]],
        name='Client'
        )
    return fig