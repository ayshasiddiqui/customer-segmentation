import streamlit as st
import pandas as pd

st.title("📊 Dashboard")

df = pd.read_excel("data/data.xlsx")

st.metric("Total Customers", df["CustomerID"].nunique())

st.metric("Total Products", df["StockCode"].nunique())

st.metric("Countries", df["Country"].nunique())

st.metric("Total Revenue",
          round((df["Quantity"]*df["UnitPrice"]).sum(),2))

st.dataframe(df.head())