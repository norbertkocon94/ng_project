import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

st.set_page_config(layout="wide", initial_sidebar_state="expanded")


def data_cleaner(df, datecolumn):
    '''

    Returns the dataframe, with no alphabetic characters, and changes the column type to float.

        Parameters:
        -----------
            df (dataframe): Dataframe
            datacolumn (str): Name of column with date

        Returns:
        -----------
            df (dataframe): Cleaned dataframe

    '''
    df.replace('([a-zA-Z])', '', regex=True, inplace=True)

    df[datecolumn] = pd.to_datetime(df[datecolumn])

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = pd.to_numeric(df[column])
        else:
            pass

    return df


def data_interpolate(df, datecolumn, method='linear'):
    '''

    Returns a dataframe. If there are NaN values ​​in the data, complete them with the interpolation method.

        Parameters:
        -----------
            df (dataframe): Dataframe
            datacolumn (str): Name of column with name
            method (str): Method of intepolation. Default - linear. See intepolation doc for more:
                          https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html

        Returns:
            df (dataframe): Dataframe with empty data filled with interpolation values

    '''
    date_time = df[datecolumn]

    df = df.iloc[:, 1:].interpolate(method=method)

    df[datecolumn] = date_time

    df = df.set_index(datecolumn, drop=True)

    return df


def data_outliers_cleaner(df, column_name, quantile, method='linear'):
    '''

    Returns a dataframe. Detection of outliers, conversion to interpolation values.

        Parameters:
        -----------
            df (dataframe): Dataframe
            datacolumn (str): Name of column with name
            method (str): Method of intepolation. Default - linear. See intepolation doc for more:
                          https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
            quantile (float): Quantiles are cut points dividing the range of a probability distribution. For more:
                          https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.quantile.html

        Returns:
            df (dataframe): Dataframe with no outliers, with interpolation values.

    '''
    if quantile > 0.80:
        df[column_name] = np.where(df[column_name] >= df[column_name].quantile(quantile), np.NaN, df[column_name])
        df = df.interpolate(method=method)
    else:
        df[column_name] = np.where(df[column_name] <= df[column_name].quantile(quantile), np.NaN, df[column_name])
        df = df.interpolate(method=method)

    return df


header = st.container()
expander = st.container()
main = st.container()
main_under = st.container()


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r'.\model.h5')
    return model


with header:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.write()

    with col2:
        # st.image(r'./jeremy-perkins-7FOSJVtUtac-unsplash.jpg')
        st.title('Market2: Open prediction app! 📈')

    with col3:
        st.write("")

with expander:
    st.sidebar.header("Sidebar expander 📝")
    st.sidebar.markdown("""
    This app is based on our Stock Market dataset.
    
    In this application I used the LSTM (Long short-term memory) model.

    - github: [Link](https://github.com/norbertkocon94/ng_project)
    - google-colab: [Link](https://colab.research.google.com/drive/1icrWF837xwGWYGx3KaSGC86SqZk5Bx2D?usp=sharing)
    ___
    """)

with main:
    col1, col3 = st.columns([1, 2])

    with col1:
        st.markdown("""
        # 1. Upload your data.
        """)

    with col3:
        st.markdown("""
        # 2. Result.
        """)

with main_under:
    col1, col3 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("Choose a .csv file!")
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            st.markdown("""
            ## Clean data.
            """)
            st.write(dataframe)

            dataframe = data_cleaner(df=dataframe, datecolumn="DateTime")
            dataframe = data_interpolate(df=dataframe, datecolumn="DateTime")
            dataframe = data_outliers_cleaner(df=dataframe, column_name='Market1: High', quantile=0.9999)
            dataframe = data_outliers_cleaner(df=dataframe, column_name='Market1: High', quantile=0.0001)
            dataframe = data_outliers_cleaner(df=dataframe, column_name='Market3: Low', quantile=0.9999)

            st.markdown("""
            ## Data after cleaning!
            """)
            st.write(dataframe)

    with col3:
        if uploaded_file is not None:
            st.image(r".\final_chart.png")
            st.image(r".\all_char.png")


