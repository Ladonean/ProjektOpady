import streamlit as st
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import geopandas as gpdr
from scipy.interpolate import griddata
from geokrige.tools import TransformerGDF
import sys
import os
import requests

#Ścieżki jako wybór trzeba zrobić oprócz tych do kształtów
#path_shapefile1 = r'C:\Users\domin\OneDrive\Pulpit\ProjektInglot\Skrypty\Mapa\gadm41_POL_0.shp'
path_shapefile1 = 'https://github.com/Ladonean/FigDetect/blob/main/gadm41_POL_1.shp'

path_csv1 = 'https://github.com/Ladonean/FigDetect/blob/main/o_d_07_2007.csv'



path_stacje1 = 'https://github.com/Ladonean/FigDetect/blob/main/Stacje.csv'
# Funkcja do wczytywania danych z pliku tekstowego

path_csv = requests.get(path_csv1)
path_shapefile = requests.get(path_shapefile1)
path_stacje = requests.get(path_stacje1)

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

def create_map():
    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=6.2)
    folium.GeoJson(gdf).add_to(m)
    folium.GeoJson(gdf1).add_to(m)
    for _, row in Wynik.iterrows():
        folium.CircleMarker(
            location=[row['X'], row['Y']],
            radius=1,
            popup=f"{row['Stacja']}: {row['Opady']} mm",
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        ).add_to(m)
    return m

def plot_wynik(path_shapefile, Wynik):

    X = np.column_stack([Wynik['X'], Wynik['Y']])
    y = np.array(Wynik['Opady'])

    prediction_gdf = gpd.read_file(path_shapefile).to_crs(crs='EPSG:4326')
    transformer = TransformerGDF()
    transformer.load(prediction_gdf)
    meshgrid = transformer.meshgrid(density=2)
    mask = transformer.mask()

    X_siatka, Y_siatka = meshgrid
    Z_siatka = griddata((X[:, 1], X[:, 0]), y, (X_siatka, Y_siatka), method='nearest')
    Z_siatka[~mask] = None

    fig, ax = plt.subplots()
    prediction_gdf.plot(facecolor='none', edgecolor='black', linewidth=1.5, zorder=5, ax=ax) 
    cbar = ax.contourf(X_siatka, Y_siatka, Z_siatka, cmap='YlGnBu', levels=np.arange(0, 360, 10), extend='min')
    cax = fig.add_axes([0.93, 0.134, 0.02, 0.72])
    colorbar = plt.colorbar(cbar, cax=cax, orientation='vertical')

    ax.grid(lw=0.2)
    ax.set_title('Opady miesiąc ...', fontweight='bold', pad=15)

    #scatter = ax.scatter(Wynik['Y'].astype(float), Wynik['X'].astype(float), c='black', marker='x',label='Punkty pomiarowe')
    
    return fig, ax

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


gdf = gpd.read_file(path_shapefile)
#gdf1 = gpd.read_file(path_shapefile1)

#m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=6.2)

#folium.GeoJson(gdf).add_to(m)
#folium.GeoJson(gdf1).add_to(m)

#nanoszenie punktów na folium
#for _, row in Wynik.iterrows():
    #folium.CircleMarker(
      #  location=[row['X'], row['Y']],
      #  radius=1,
      #  popup=f"{row['Stacja']}: {row['Opady']} mm",
      #  color='blue',
      #  fill=True,
       # fill_color='blue',
       # fill_opacity=0.6
   # ).add_to(m)

st.title('Tabela')

#st_folium(m, width=700, height=600)

st.dataframe(Wynik, width=800, height=1200)





