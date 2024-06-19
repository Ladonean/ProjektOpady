import streamlit as st
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import requests

#Ścieżki jako wybór trzeba zrobić oprócz tych do kształtów

path_csv1 = 'https://github.com/Ladonean/FigDetect/blob/main/o_d_07_2007.csv'



path_stacje1 = 'https://github.com/Ladonean/FigDetect/blob/main/Stacje.csv'
# Funkcja do wczytywania danych z pliku tekstowego

path_csv = requests.get(path_csv1)
path_csv = StringIO(path_csv.text)
path_stacje = requests.get(path_stacje1)
path_stacje = StringIO(path_stacje.text)


def wczytaj_csv(path_csv):

    box=[]

    with open(path_csv, 'r') as file:
        csvreader = csv.reader(file)

        for row in csvreader:
            box.append(row)

    file_csv = np.array(box)
    file_csv = file_csv.reshape(-1, 16)
    col_0 = file_csv[:, 0]
    col_1 = file_csv[:, 1]
    col_2 = file_csv[:, 5]

    # Łączenie kolumn w nową macierz
    file_csv = np.column_stack((col_0, col_1, col_2))

    return file_csv

def wczytaj_stacje(path_stacje):

    box=[]

    with open(path_stacje, 'r') as file:
        csvreader = csv.reader(file)

        for row in csvreader:
            box.append(row)

    Lista = np.array(box)
    Lista = Lista.reshape(-1, 3)

    Lista = np.column_stack((Lista[:,0], Lista[:,1], Lista[:,2]))

    #Lista[:,0], Lista[:,1] = transformacja.transform(Lista[:,0], Lista[:,1])
    Lista = pd.DataFrame(Lista, columns=['X','Y', 'Stacja'])
    Lista['X'] = Lista['X'].astype(float)
    Lista['Y'] = Lista['Y'].astype(float)
    return Lista

def suma_opadów(tabela):
    df = pd.DataFrame(tabela, columns=['Kod stacji','Stacja', 'Opady'])
    df['Opady'] = pd.to_numeric(df['Opady'])

    df_suma = df.groupby('Stacja')['Opady'].sum().reset_index()
    df_suma['Opady'] = df_suma['Opady'].astype(float)
    return df_suma




st.title("OpadyPolska")

df = wczytaj_csv(path_csv)

df_suma = suma_opadów(df)

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





