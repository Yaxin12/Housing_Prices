import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.header('Visualizations of House Prices')

def load_data(csv):
    df = pd.read_csv(csv)
    return df

project = load_data('hpi-2.csv')
st.dataframe(project)
