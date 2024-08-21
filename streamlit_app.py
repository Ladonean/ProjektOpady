#Biblioteki
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
import geopandas as gpd
from geokrige.tools import TransformerGDF
import calendar
from scipy.interpolate import Rbf

# Tłumaczenie polskich nazw miesięcy na angielskie
miesiac_d = {
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

# Wczytanie csv ze opadami
def wczytaj_csv(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Nie udało się pobrać danych z podanego URL: " + url)
        return None
    data = response.content.decode('windows-1250')
    df = pd.read_csv(StringIO(data), delimiter=',', header=None)
    
    df = df.iloc[:, [1, 5]]
    df.columns = ['Stacja', 'Opady']
    
    return df

# Wczytanie csv ze stacjami
def wczytaj_stacje(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Nie udało się pobrać danych stacji z podanego URL: " + url)
        return None
    data = response.content.decode('windows-1250')
    df = pd.read_csv(StringIO(data), delimiter=',', header=None)
    
    df.columns = ['X', 'Y', 'Stacja']
    df['X'] = df['X'].astype(float)
    df['Y'] = df['Y'].astype(float)
    
    return df

# Obliczanie sumy opadów w miesiącu
def suma_opadow(tabela):
    if tabela is None:
        return None
    tabela['Opady'] = pd.to_numeric(tabela['Opady'], errors='coerce')
    df_suma = tabela.groupby('Stacja')['Opady'].sum().reset_index()
    df_suma['Opady'] = df_suma['Opady'].astype(float)
    return df_suma

def plot_wynik(path_shp, Wynik, title):
    X = np.column_stack([Wynik['X'], Wynik['Y']])
    y = np.array(Wynik['Opady'])

    granica = gpd.read_file(path_shp).to_crs(crs='EPSG:4326')
    transformer = TransformerGDF()
    transformer.load(granica)
    meshgrid = transformer.meshgrid(density=3)
    mask = transformer.mask()

    X_siatka, Y_siatka = meshgrid

    # Aproksymacja wielomianowa z wygładzaniem 1
    rbf_interpolator = Rbf(X[:, 1], X[:, 0], y, function='multiquadric', smooth=1)
    Z_siatka = rbf_interpolator(X_siatka, Y_siatka)
    Z_siatka[~mask] = None

    fig, ax = plt.subplots()
    granica.plot(facecolor='none', edgecolor='black', linewidth=1.5, zorder=5, ax=ax)
    y_s = np.sort(y)[-5:]
    avg5 = np.mean(y_s)
    cbar = ax.contourf(X_siatka, Y_siatka, Z_siatka, cmap='YlGnBu', levels=np.arange(0, avg5, 20), extend='min')
    cax = fig.add_axes([0.93, 0.134, 0.02, 0.72])
    colorbar = plt.colorbar(cbar, cax=cax, orientation='vertical')

    ax.grid(lw=0.3)
    ax.set_title(title, fontweight='bold', pad=15)

    return fig, ax

# Streamlit tytuł
st.title("Opady Polska")

# Wybór roku i miesiąca
rok = st.selectbox("Wybierz rok", [2010,2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
miesiac = st.selectbox("Wybierz miesiąc", 
                        ["Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec", 
                        "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień"])

# Konwersja nazwy miesiąca na dwucyfrowy numer miesiąca
e_miesiac = miesiac_d[miesiac]
miesiac_l = str(list(calendar.month_name).index(e_miesiac)).zfill(2)

# Generowanie ścieżki pliku na podstawie wyboru
path_opady = f'https://raw.githubusercontent.com/Ladonean/FigDetect/main/o_d_{miesiac_l}_{rok}.csv'
path_stacje = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/Stacje.csv'
path_shp = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/gadm41_POL_1.shp'

# Wczytywanie danych
df_opady = wczytaj_csv(path_opady)
df_stacje = wczytaj_stacje(path_stacje)

if df_opady is not None and df_stacje is not None:
    df_suma = suma_opadow(df_opady)
    
    # Łączenie danych
    df_stacje['Stacja'] = df_stacje['Stacja'].str.strip()
    df_suma['Stacja'] = df_suma['Stacja'].str.strip()
    
    Wynik = pd.merge(df_stacje, df_suma[['Stacja', 'Opady']], on='Stacja', how='left')
    Wynik = Wynik.dropna()
    
    # Wyświetlanie danych
    st.title('Tabela Stacje')
    st.dataframe(Wynik, width=800, height=1200)
    
    max_value = Wynik['Opady'].astype(float).max()
    min_value = Wynik['Opady'].astype(float).min()
    
    st.write(f"Maksymalna ilość opadów: {max_value}")
    st.write(f"Minimalna ilość opadów: {min_value}")

    # Rysowanie mapy
    fig, ax = plot_wynik(path_shp, Wynik, f'Opady {miesiac} {rok}')
    st.title('Mapa Opadów')
    st.pyplot(fig)
else:
    st.error("Nie udało się załadować danych.")

