import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import geopandas as gpd
from geokrige.tools import TransformerGDF
from scipy.interpolate import griddata
import calendar

# Tłumaczenie polskich nazw miesięcy na angielskie
months_dict = {
    "Styczeń": "January",
    "Luty": "February",
    "Marzec": "March",
    "Kwiecień": "April",
    "Maj": "May",
    "Czerwiec": "June",
    "Lipiec": "July",
    "Sierpień": "August",
    "Wrzesień": "September",
    "Październik": "October",
    "Listopad": "November",
    "Grudzień": "December"
}

# Function to read CSV from URL
def wczytaj_csv(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Nie udało się pobrać danych z podanego URL")
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
        st.error("Nie udało się pobrać danych stacji z podanego URL")
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
    ax.set_title(f'Opady {month} {year}', fontweight='bold', pad=15)

    return fig, ax

# Streamlit app layout
st.title("OpadyPolska")

# Wybór roku i miesiąca
year = st.selectbox("Wybierz rok", [2018, 2019, 2020, 2021, 2022])
month = st.selectbox("Wybierz miesiąc", 
                     ["Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec", 
                      "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień"])

# Konwersja nazwy miesiąca na dwucyfrowy numer miesiąca
english_month = months_dict[month]
month_number = str(list(calendar.month_name).index(english_month)).zfill(2)

# Generowanie ścieżki pliku na podstawie wyboru
path_csv1 = f'https://raw.githubusercontent.com/Ladonean/FigDetect/main/o_d_{month_number}_{year}.csv'
path_stacje1 = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/Stacje.csv'
path_shapefile = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/gadm41_POL_1.shp'

# Fetching and processing data
df = wczytaj_csv(path_csv1)
df_baza = wczytaj_stacje(path_stacje1)

if df is not None and df_baza is not None:
    df_suma = suma_opadow(df)
    
    # Łączenie danych
    df_baza['Stacja'] = df_baza['Stacja'].str.strip()
    df_suma['Stacja'] = df_suma['Stacja'].str.strip()
    
    Wynik = pd.merge(df_baza, df_suma[['Stacja', 'Opady']], on='Stacja', how='left')
    Wynik = Wynik.dropna()
    
    # Wyświetlanie danych
    st.title('Tabela')
    st.dataframe(Wynik, width=800, height=1200)
    
    max_value = Wynik['Opady'].astype(float).max()
    min_value = Wynik['Opady'].astype(float).min()
    
    st.write(f"Maksymalna ilość opadów: {max_value}")
    st.write(f"Minimalna ilość opadów: {min_value}")

    # Rysowanie mapy
    fig, ax = plot_wynik(path_shapefile, Wynik)
    st.pyplot(fig)

else:
    st.error("Nie udało się załadować danych.")
