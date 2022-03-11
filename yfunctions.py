# -*- coding: utf-8 -*-
"""
Created:01/03/2022 10:13

@author: Yannick Le Hellaye
"""
### Declaration ###
import pickle
import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
import base64
#import shap
import plotly.graph_objects as go
import plotly.express as px

def gauge_charts(xval,yseuil):

    fig = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = xval,
        mode = "gauge+number+delta",
        title = {'text': "Score"},
        delta = {'reference': yseuil},
        gauge = {'axis': {'range': [None, 1.0]},
                 'steps' : [
                     {'range': [0, 0.5], 'color': "lightgray"},
                     {'range': [0.5, 1.0], 'color': "gray"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': yseuil}}))

    return fig

