import time
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
from sklearn.metrics import mean_squared_error
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from prophet.diagnostics import cross_validation 
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
import plotly.graph_objs as go
import plotly
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt 
from dateutil.relativedelta import relativedelta
import numpy as np

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
app_mode = st.sidebar.selectbox('🔎 Select Page',['02 Prediction 🎯','01 Introduction 🚀'])
select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Train","Test"])
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
        alt.Chart(data, title="Evolution of Decathlon Turnover for this specific Department")
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

@st.cache_data(ttl=60 * 60 * 24)
def comparison_chart(data, title):
    hover = alt.selection_single(
        fields=["date"],
        nearest=True,
        on="mouseover",
        empty="none",
    )

    lines = (
        alt.Chart(data, title=title)
        .mark_line()
        .encode(
            x="date",
            y='sales:Q',
            color="set",
        )
    )
    # Draw points on the line, and highlight based on selection
    points = lines.transform_filter(hover).mark_circle(size=65)

    # Draw a rule at the location of the selection
    tooltips = (
        alt.Chart(data)
        .mark_rule()
        .encode(
            x="date",
            y='sales:Q',
            opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
            tooltip=[
                alt.Tooltip("date", title="Date"),
                alt.Tooltip('sales:Q', title=["Sales ($)"]),
            ],
        )
        .add_selection(hover)
    )

    return (lines + points + tooltips).interactive()

# functions
@st.cache_data
def get_simple_model(df):
	# df.columns = ['ds','y']
	m = Prophet(interval_width=0.95)
	m.fit(df) 
	return m 

@st.cache_data
def get_fc_charts(df, start_date, periods):
	st.write('#### Model predictions')
	df.columns = ['ds','y']
	m = get_simple_model(df)
	future = m.make_future_dataframe(periods=periods, freq = 'MS')
	forecast = m.predict(future)

	# evaluate the predictions
	st.write('#### Evaluate model predictions')
	dates = df['ds'][-periods:].values
	y_true = df['y'][-periods:].values
	y_pred = forecast['yhat'][-periods:].values

	# assess the model with MAE
	mae = mean_absolute_error(y_true, y_pred)
	st.success('MAE: %.3f' % mae)


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
st.image(image_edhec, width=200)






if select_dataset == "Train":
    df = pd.read_csv("datasets/train_merged.csv")
    st.success(f'You have selected {"Train Dataset"}. Here are the top 5 rows from dataset')
else:
    df = pd.read_csv(".datasets/test_merged.csv")
    st.success(f'You have selected {"Test Dataset"}. Here are the top 5 rows from dataset')


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



if app_mode == '02 Prediction 🎯':
    df["date"] = df["day_id"]
    st.subheader(" ")
    st.subheader("01 - Show  Decathlon Time Series ")
    st.subheader(" ")

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

    df["date"] = dates

    df = df[df['date'] > pd.to_datetime(start_date)]


    df_new = df[df["turnover"]<1000000].sample(n=10000).reset_index(drop=True)
    list_dep = list(df_new["dpt_num_department"].unique())
    depart_val = st.selectbox("Select a Department Number", [127,88,117,73])
    #st.write(depart_val)
    df_new_store = df_new[df_new["dpt_num_department"] == depart_val]
    chart = get_chart(df_new_store)
    st.altair_chart((chart).interactive(), use_container_width=True)
    df_new_store_group = df_new_store.groupby("date").agg({'turnover': 'sum'}).reset_index()
    chart = get_chart(df_new_store_group)
    st.altair_chart((chart).interactive(), use_container_width=True)



    st.subheader(" ")
    st.subheader(f"04 - Decomposition of the {depart_val} Department Time Serie")
    st.subheader(" ")

    start_time = time.time()
    df_new_store_group.index = pd.to_datetime(df_new_store_group["date"])
    df_new_store_group_v2 = df_new_store_group["turnover"]

    df_new_store_group_v2.index = pd.to_datetime(df_new_store_group_v2.index)

    plt.rc('figure',figsize=(14,8))
    plt.rc('font',size=15)
    result = seasonal_decompose(df_new_store_group_v2,model='additive', period=12)
    fig = result.plot()
    st.pyplot()


    st.write('#### Model predictions')
    df_new_store_group.columns = ['ds','y']
    st.sidebar.write('#### Hyperparameter Tuning 🔬')
    select_growth = st.sidebar.selectbox("1️⃣ Select Growth Type:",["linear","logistic"])
    select_seasonality_mode = st.sidebar.selectbox("2️⃣ Select Seasonality Mode:",["additive","multiplicative"])
    select_changepoint_prior_scale = st.sidebar.slider("3️⃣ Select Flexibility of trend:",0.01,0.10,0.05,0.01)
    m = Prophet(growth=select_growth,seasonality_mode=select_seasonality_mode,
                changepoint_prior_scale=select_changepoint_prior_scale,)
    m.fit(df_new_store_group)

    # Make a forecast
    future = m.make_future_dataframe(periods=8*7, freq='D')
    forecast = m.predict(future)


    # Plot the forecast
    periods = 8*7
    y_true = df_new_store_group['y'][-periods:].values
    y_pred = forecast['yhat'][247-periods:247].values
    dates = forecast['ds'][247-periods:247].values

    fig, ax = plt.subplots()
    ax.plot(dates, y_true, label="Real")
    ax.plot(dates, y_pred, label="Prediction")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Turnover")
    ax.set_title("Real Values vs Prediction - Turnover")
    ax.legend()
    st.pyplot(fig)

    y_pred2 = forecast['yhat'][247:].values
    dates2 = forecast['ds'][247:].values

    fig2, ax2 = plt.subplots()
    ax2.plot(dates2, y_pred2, label="Prediction")
    ax.set_xlabel("Dates")
    ax.set_ylabel("Prediction")
    ax.set_title("Forecast 8 weeks - Turnover")
    ax.legend()
    st.pyplot(fig2)

    # assess the model with MAE
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    st.markdown("#### Performance metrics 🚀")
    st.success('MAE: %.3f' % mae)
    st.success('MSE: %.3f' % mse)
    st.success('RMSE: %.3f' % rmse)

    st.markdown("#### Execution Time Model ⚙️")

    st.warning("--- %s seconds ---" % (np.round(time.time() - start_time,2)))

    st.markdown("#### Sustainable metrics 🌱")

    from codecarbon import OfflineEmissionsTracker
    tracker = OfflineEmissionsTracker(country_iso_code="FRA") # FRA = France
    tracker.start()
    results = tracker.stop()
    st.error(' %.12f kWh' % results)

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

