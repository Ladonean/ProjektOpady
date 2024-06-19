import streamlit as st
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import requests
from io import StringIO

#Ścieżki jako wybór trzeba zrobić oprócz tych do kształtów

path_csv1 = 'https://github.com/Ladonean/FigDetect/blob/main/o_d_07_2007.csv'



path_stacje1 = 'https://github.com/Ladonean/FigDetect/blob/main/Stacje.csv'
# Funkcja do wczytywania danych z pliku tekstowego

path_csv = requests.get(path_csv1)

path_stacje = requests.get(path_stacje1)



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

def suma_opadow(tabela):
    
    tabela['Opady'] = pd.to_numeric(tabela['Opady'], errors='coerce')
    df_suma = tabela.groupby('Stacja')['Opady'].sum().reset_index()
    df_suma['Opady'] = df_suma['Opady'].astype(float)
    
    return df_suma





st.title("OpadyPolska")

df = wczytaj_csv(path_csv)

df_suma = suma_opadow(df)

df_baza = wczytaj_stacje(path_stacje)

# Łączenie DataFrame'ów po kolumnie 'Stacja'
df_baza['Stacja'] = df_baza['Stacja'].str.strip()
df_suma['Stacja'] = df_suma['Stacja'].str.strip()

Wynik = pd.merge(df_baza, df_suma[['Stacja', 'Opady']], on='Stacja', how='left')
Wynik = Wynik.dropna()

max_value = Wynik['Opady'].astype(float).max()
min_value = Wynik['Opady'].astype(float).min()




st.title('Tabela')


st.dataframe(Wynik, width=800, height=1200)





