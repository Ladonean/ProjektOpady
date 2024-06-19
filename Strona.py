import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO

# URLs to the CSV files
path_csv1 = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/o_d_07_2007.csv'
path_stacje1 = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/Stacje.csv'

# Function to read CSV from URL
def wczytaj_csv(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Could not fetch CSV data from the provided URL")
        return None
    data = response.content.decode('windows-1250')
    df = pd.read_csv(StringIO(data), delimiter=',', header=None)
    
    # Selecting and reshaping relevant columns
    df = df.iloc[:, [0, 1, 5]]
    df.columns = ['Kod stacji', 'Stacja', 'Opady']
    
    return df

# Function to read station data from URL
def wczytaj_stacje(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Could not fetch station data from the provided URL")
        return None
    data = response.content.decode('windows-1250')
    df = pd.read_csv(StringIO(data), delimiter=',', header=None)
    
    df.columns = ['X', 'Y', 'Stacja']
    df['X'] = df['X'].astype(float)
    df['Y'] = df['Y'].astype(float)
    
    return df

# Function to calculate sum of rainfall
def suma_opadow(tabela):
    tabela['Opady'] = pd.to_numeric(tabela['Opady'], errors='coerce')
    df_suma = tabela.groupby('Stacja')['Opady'].sum().reset_index()
    df_suma['Opady'] = df_suma['Opady'].astype(float)
    return df_suma

# Streamlit app layout
st.title("OpadyPolska")

# Fetching and processing data
df = wczytaj_csv(path_csv1)
df_baza = wczytaj_stacje(path_stacje1)

if df is not None and df_baza is not None:
    df_suma = suma_opadow(df)
    
    # Merging data
    df_baza['Stacja'] = df_baza['Stacja'].str.strip()
    df_suma['Stacja'] = df_suma['Stacja'].str.strip()
    
    Wynik = pd.merge(df_baza, df_suma[['Stacja', 'Opady']], on='Stacja', how='left')
    Wynik = Wynik.dropna()
    
    # Displaying data
    st.title('Tabela')
    st.dataframe(Wynik, width=800, height=1200)
    
    max_value = Wynik['Opady'].astype(float).max()
    min_value = Wynik['Opady'].astype(float).min()
    
    st.write(f"Max Opady: {max_value}")
    st.write(f"Min Opady: {min_value}")

else:
    st.error("Failed to load data.")
