import altair as alt
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
from PIL import Image, ImageDraw, ImageFont

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

from pathlib import Path
import base64
import time
from datetime import date, datetime
from pandas import read_csv
from pandas import to_datetime
from pandas import DataFrame
from prophet import Prophet
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
import seaborn as sns


sns.set(style="whitegrid")
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
st.set_option('deprecation.showPyplotGlobalUse', False)











st.set_page_config(
    page_title="Decathlon - Predictive Analytics App ", layout="wide", page_icon="./images/flask.png"
)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('🔎 Select Page',['01 Introduction 🚀','02 Visualization 📊','03 Prediction 🎯'])
select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Train","Test","Bu feat","Merged"])
list_kpi = ['turnover']
kpi = st.sidebar.selectbox("📈 Select KPI", list_kpi)

@st.cache_data(ttl=60 * 60 * 24)
def get_chart(data):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title="Evolution of stock prices")
        .mark_line()
        .encode(
            x="date",
            y=kpi,
            #color="symbol",
            # strokeDash="symbol",
        )
    )

    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="yearmonthdate(date)",
            y=kpi,
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip(kpi, title="Price (USD)"),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

image_edhec = Image.open('images/decathlon.png')
st.image(image_edhec, width=250)






if select_dataset == "Train":
    df = pd.read_csv("datasets/train.csv")
    st.success(f'You have selected {"Train Dataset"}. Here are the top 5 rows from dataset')
elif select_dataset == "Test": 
    df = pd.read_csv(".datasets/test.csv")
    st.success(f'You have selected {"Test Dataset"}. Here are the top 5 rows from dataset')
elif select_dataset == "Bu feat": 
    df = pd.read_csv("datasets/bu_feat.csv")
    st.success(f'You have selected {"Bu feat Dataset"}. Here are the top 5 rows from dataset')
else: 
    df = pd.read_csv("datasets/merged.csv")
    st.success(f'You have selected {"Merged Dataset"}. Here are the top 5 rows from dataset')


if app_mode == '01 Introduction 🚀':

    st.markdown("### 00 - Show  Dataset")
    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))
    
    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    st.text('(Rows,Columns)')
    st.write(df.shape)


    st.markdown("### 01 - Description")
    st.dataframe(df.describe())



    st.markdown("### 02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("Looks good! as we have less then 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    st.markdown("### 03 - Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("Looks good! as we have completeness ratio greater than 0.85.")
           
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")

    st.markdown("### 04 - Complete Report")
    if st.button("Generate Report"):

        pr = df.profile_report()
        #st_profile_report(pr)



if app_mode == '03 Prediction 🎯':
    df["date"] = df["day_id"]
    print("Hello World")

    start_date = st.date_input(
        "Select start date",
        date(2013, 1, 1),
        min_value=datetime.strptime("2013-01-01", "%Y-%m-%d"),
        max_value=datetime.now(),
    )   

    # example list of strings representing dates
    date_strings = list(df["date"])

    # empty list to store converted datetime dates
    dates = []

    # loop through each string and convert to datetime date
    for date_str in date_strings:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        dates.append(date)

    # print the list of converted dates
    #print(dates)
    df["date"] = dates
    #st.write(type(start_date))
    #st.write(type(df["day_id"][0]))
    #st.dataframe(df_master)
    df = df[df['date'] > pd.to_datetime(start_date)]

    st.subheader(" ")
    st.subheader("01 - Show  Selected Stocks Time Series ")
    st.subheader(" ")
    df_new = df[df["turnover"]<1000000].reset_index(drop=True)
    df_new_store = df_new[df_new["dpt_num_department"] == 127]
    chart = get_chart(df_new_store)
    st.altair_chart((chart).interactive(), use_container_width=True)



if __name__=='__main__':
    main()

st.markdown(" ")
st.markdown("### 👨🏼‍💻 **App Contributors:** ")
st.image(['images/gaetan.png'], width=100,caption=["Gaëtan Brison"])

st.markdown(f"####  Link to Project Website [here]({'https://github.com/gaetanbrison/app-predictive-analytics'}) 🚀 ")
st.markdown(f"####  Feel free to contribute to the app and give a ⭐️")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        " Made by ",
        link("https://github.com/gaetanbrison", "Gaëtan Brison"),
        "👨🏼‍💻"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()

