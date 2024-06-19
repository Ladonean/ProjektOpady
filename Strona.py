import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import geopandas as gpd
from geokrige.tools import TransformerGDF
from scipy.interpolate import griddata

# URLs to the CSV files
path_csv1 = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/o_d_07_2007.csv'
path_stacje1 = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/Stacje.csv'
path_shapefile = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/gadm41_POL_1.shp'

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


fig, ax = plot_wynik(path_shapefile, Wynik)

st.pyplot(fig)
