import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)
st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

/* Main Background */
.stApp{
    background:#F4F7FC;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0083B0,#00B4DB);
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Headings */
h1{
    color:#0083B0;
    font-weight:700;
}

h2,h3{
    color:#005F73;
}

/* Paragraph */
p, label{
    color:#333333;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:18px;
    border-left:5px solid #00B4DB;
    box-shadow:0px 5px 15px rgba(0,0,0,0.12);
}

/* DataFrame */
[data-testid="stDataFrame"]{
    background:white;
    border-radius:12px;
    box-shadow:0px 5px 15px rgba(0,0,0,0.12);
}

/* Buttons */
.stButton>button,
.stDownloadButton>button{
    background:#00B4DB;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover,
.stDownloadButton>button:hover{
    background:#0083B0;
    color:white;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"]{
    background:white;
    color:black;
    border-radius:10px;
}

/* Radio */
div[role="radiogroup"]{
    background:white;
    border-radius:10px;
    padding:10px;
}

/* Tables */
table{
    color:black;
}

/* Horizontal Line */
hr{
    border:1px solid #D0D7DE;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load Dataset
# ------------------------------

@st.cache_data
def load_data():

    df = pd.read_excel("data.xlsx")

    # Data Cleaning

    df = df.drop_duplicates()

    df = df.dropna(subset=["CustomerID"])

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["CustomerID"] = df["CustomerID"].astype(int)

    df["StockCode"] = df["StockCode"].astype(str)

    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df


df = load_data()

# ------------------------------
# Load Model
# ------------------------------

model = joblib.load("model.pkl")

scaler = joblib.load("scaler.pkl")

with open("cluster_info.json","r") as file:
    cluster_info = json.load(file)

# ------------------------------
# Sidebar
# ------------------------------

st.sidebar.title("🛒 Customer Segmentation")

page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Home",

        "📂 Dataset",

        "📊 Dashboard",

        "🎯 Prediction"

    ]

)

# ==========================================================
# HOME PAGE
# ==========================================================

if page=="🏠 Home":

    st.title("🛒 Customer Segmentation using Machine Learning")

    st.write("""

    ## Welcome

    This project is developed using

    ✅ Python

    ✅ Data Science

    ✅ Machine Learning

    ✅ KMeans Clustering

    """)

    st.markdown("---")

    c1,c2,c3,c4=st.columns(4)

    c1.metric(

        "Customers",

        df["CustomerID"].nunique()

    )

    c2.metric(

        "Countries",

        df["Country"].nunique()

    )

    c3.metric(

        "Orders",

        df["InvoiceNo"].nunique()

    )

    c4.metric(

        "Products",

        df["Description"].nunique()

    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(

        df.head(10),

        use_container_width=True

    )

# ==========================================================
# DATASET PAGE
# ==========================================================

elif page=="📂 Dataset":

    st.title("Dataset")

    st.subheader("Dataset Preview")

    st.dataframe(

        df,

        use_container_width=True

    )

    st.markdown("---")

    st.subheader("Shape")

    st.write(df.shape)

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    missing = pd.DataFrame({

        "Column":df.columns,

        "Missing Values":df.isnull().sum().values

    })

    st.dataframe(missing)

    st.markdown("---")

    st.subheader("Statistics")

    st.dataframe(

        df.describe()

    )
    # ==========================================================
# DASHBOARD PAGE
# ==========================================================

elif page == "📊 Dashboard":

    st.title("📊 Customer Dashboard")

    st.markdown("---")

    # ================= KPIs =================

    total_sales = df["TotalPrice"].sum()

    total_customers = df["CustomerID"].nunique()

    total_orders = df["InvoiceNo"].nunique()

    total_products = df["Description"].nunique()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("💰 Total Sales", f"₹ {total_sales:,.2f}")

    c2.metric("👤 Customers", total_customers)

    c3.metric("🛒 Orders", total_orders)

    c4.metric("📦 Products", total_products)

    st.markdown("---")

    # ================= Country Filter =================

    country = st.selectbox(

        "Select Country",

        ["All"] + sorted(df["Country"].dropna().unique().tolist())

    )

    if country == "All":

        filtered_df = df.copy()

    else:

        filtered_df = df[df["Country"] == country]

    st.success(f"Showing data for : {country}")

    st.markdown("---")

    # ================= Top Countries =================

    st.subheader("🌍 Top 10 Countries")

    country_sales = (

        filtered_df

        .groupby("Country")["TotalPrice"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

    )

    fig, ax = plt.subplots(figsize=(10,5))

    country_sales.plot(

        kind="bar",

        ax=ax,

        color="skyblue"

    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("---")

    # ================= Monthly Sales =================

    st.subheader("📈 Monthly Sales")

    monthly_sales = (

        filtered_df

        .groupby(filtered_df["InvoiceDate"].dt.to_period("M"))["TotalPrice"]

        .sum()

    )

    monthly_sales.index = monthly_sales.index.astype(str)

    fig, ax = plt.subplots(figsize=(12,5))

    monthly_sales.plot(

        marker="o",

        ax=ax

    )

    plt.xticks(rotation=45)

    st.pyplot(fig)

    st.markdown("---")

    # ================= Country Distribution =================

    st.subheader("🥧 Country Distribution")

    country_count = (

        filtered_df["Country"]

        .value_counts()

        .head(10)

    )

    fig, ax = plt.subplots(figsize=(7,7))

    ax.pie(

        country_count,

        labels=country_count.index,

        autopct="%1.1f%%"

    )

    st.pyplot(fig)

    st.markdown("---")

    # ================= Top Products =================

    st.subheader("📦 Top Selling Products")

    top_products = (

        filtered_df

        .groupby("Description")["Quantity"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

    )

    fig, ax = plt.subplots(figsize=(12,5))

    top_products.plot(

        kind="bar",

        ax=ax,

        color="orange"

    )

    plt.xticks(rotation=90)

    st.pyplot(fig)

    st.markdown("---")

    # ================= Download Button =================

    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="⬇ Download Dataset",

        data=csv,

        file_name="customer_dataset.csv",

        mime="text/csv"

    )