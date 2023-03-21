import streamlit as st
import pandas as pd
import os

# Data Profiling 도구 import
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# ML 도구들 import

from pycaret.classification import setup, compare_models, pull, save_model


with st.sidebar:
    st.image("https://www.zdnet.com/a/img/resize/28d3f004b94bd200eea41399d2be2fff7505906a/2018/04/13/36c52953-7ab9-4608-a848-71d1d538856e/td-deep-learning.jpg?auto=webp&fit=crop&height=675&width=1200")
    st.title("AutoStreamML")
    choice = st.radio("Navigation", ["Upload","Profiling","ML","Download"])
    st.info("This application allows you to build an automated ML pipline using Streamlit, Pandas Profiling and PyCaret.")

if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

if choice == "Upload":
    st.title("Upload Your Data for Modelling")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df) 
        # do something 
        pass

if choice == "Profiling":
    st.title("Automated Exproratory Data Analysis")
    profile_report =df.profile_report()
    st_profile_report(profile_report)



if choice == "ML":
    st.title("Machine learning go ~~~")
    target = st.selectbox("Select Your Target", df.columns)
    if st.button("Train model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("This is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("This is the ML model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')


if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the Model", f, "trained_model.pkl")
    pass


