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
from sklearn.ensemble import RandomForestRegressor

# Tłumaczenie polskich nazw miesięcy na angielskie
mies_d = {
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

# Odczytywanie z csv
def wczytaj_csv(url):
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Nie udało się pobrać danych z podanego URL: " + url)
        return None
    data = response.content.decode('windows-1250')
    df = pd.read_csv(StringIO(data), delimiter=',', header=None)
    
    # Wyciąganie danych
    df = df.iloc[:, [0, 1, 5]]
    df.columns = ['Kod stacji', 'Stacja', 'Opady']
    
    return df

# wczytanie z githuba
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

# suma opadów dla miesiąca
def suma_opadow(tabela):
    if tabela is None:
        return None
    tabela['Opady'] = pd.to_numeric(tabela['Opady'], errors='coerce')
    df_suma = tabela.groupby('Stacja')['Opady'].sum().reset_index()
    df_suma['Opady'] = df_suma['Opady'].astype(float)
    return df_suma

def plot_wynik(shp, Wynik, title):
    X = np.column_stack([Wynik['X'], Wynik['Y']])
    y = np.array(Wynik['Opady'])

    gdf_p = gpd.read_file(shp).to_crs(crs='EPSG:4326')
    transformer = TransformerGDF()
    transformer.load(gdf_p)
    meshgrid = transformer.meshgrid(density=2)
    maska = transformer.maska()

    X_siatka, Y_siatka = meshgrid
    Z_siatka = griddata((X[:, 1], X[:, 0]), y, (X_siatka, Y_siatka), method='nearest')
    Z_siatka[~maska] = None

    fig, ax = plt.subplots()
    gdf_p.plot(facecolor='none', edgecolor='black', linewidth=1.5, zorder=5, ax=ax) 
    cbar = ax.contourf(X_siatka, Y_siatka, Z_siatka, cmap='YlGnBu', levels=np.arange(0, 360, 10), extend='min')
    cax = fig.add_axes([0.93, 0.134, 0.02, 0.72])
    colorbar = plt.colorbar(cbar, cax=cax, orientation='vertical')

    ax.grid(lw=0.2)
    ax.set_title(title, fontweight='bold', pad=15)

    return fig, ax

def predict_rainfall(X_train, y_train, X_pred):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)
    return y_pred

# Streamlit app layout
st.title("OpadyPolska")

# Wybór roku i miesiąca
rok = st.selectbox("Wybierz rok", [2010,2011,2012,2013,2014,2015,2016,2017,2018, 2019, 2020, 2021, 2022])
mies = st.selectbox("Wybierz miesiąc", 
                     ["Styczeń", "Luty", "Marzec", "Kwiecień", "Maj", "Czerwiec", 
                      "Lipiec", "Sierpień", "Wrzesień", "Październik", "Listopad", "Grudzień"])

# Konwersja nazwy miesiąca na dwucyfrowy numer miesiąca
english_mies = mies_d[mies]
mies_l = str(list(calendar.mies_name).index(english_mies)).zfill(2)

# Generowanie ścieżki pliku na podstawie wyboru
path_csv1 = f'https://raw.githubusercontent.com/Ladonean/FigDetect/main/o_d_{mies_l}_{rok}.csv'
path_stacje1 = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/Stacje.csv'
shp = 'https://raw.githubusercontent.com/Ladonean/FigDetect/main/gadm41_POL_1.shp'

# procesowanie danych
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
    
    max = Wynik['Opady'].astype(float).max()
    min = Wynik['Opady'].astype(float).min()
    
    st.write(f"Maksymalna ilość opadów: {max}")
    st.write(f"Minimalna ilość opadów: {min}")

    # Rysowanie mapy
    fig, ax = plot_wynik(shp, Wynik, f'Opady {mies} {rok}')
    st.pyplot(fig)
else:
    st.error("Nie udało się załadować danych.")

# Przewidywanie
st.title("Przewidywanie Opadów")

# Wybór zakresu dat do przewidywania
start_rok = st.selectbox("Wybierz rok początkowy", [2010,2011,2012,2013,2014,2015,2016,2017,2018, 2019, 2020, 2021])
end_rok = st.selectbox("Wybierz rok końcowy", [2011,2012,2013,2014,2015,2016,2017,2018, 2019, 2020, 2021, 2022])
start_mies = st.selectbox("Wybierz miesiąc początkowy", list(mies_d.keys()))
end_mies = st.selectbox("Wybierz miesiąc końcowy", list(mies_d.keys()))

# Wybór docelowej daty do przewidywania
pred_rok = st.selectbox("Wybierz rok do przewidywania", [2024, 2025, 2026])
pred_mies = st.selectbox("Wybierz miesiąc do przewidywania", list(mies_d.keys()))

if st.button("Przewiduj"):
    # Zbieranie danych z wybranego zakresu
    all_data = []
    for yr in range(start_rok, end_rok + 1):
        for miess in mies_d.keys():
            mies_l = str(list(calendar.mies_name).index(mies_d[miess])).zfill(2)
            path_csv = f'https://raw.githubusercontent.com/Ladonean/FigDetect/main/o_d_{mies_l}_{yr}.csv'
            df_temp = wczytaj_csv(path_csv)
            if df_temp is not None:
                df_temp['Rok'] = yr
                df_temp['Miesiąc'] = mies_l
                all_data.append(df_temp)

    if all_data:
        df_all = pd.concat(all_data)
        df_all['Opady'] = pd.to_numeric(df_all['Opady'], errors='coerce')
        
        # Suma opadów miesięcznych
        df_all_suma = df_all.groupby(['Rok', 'Miesiąc', 'Stacja']).sum().reset_index()
        
        df_baza['Stacja'] = df_baza['Stacja'].str.strip()
        df_all_suma['Stacja'] = df_all_suma['Stacja'].str.strip()
        
        Wynik_all = pd.merge(df_baza, df_all_suma[['Stacja', 'Opady', 'Rok', 'Miesiąc']], on='Stacja', how='left')
        Wynik_all = Wynik_all.dropna()

        # Przygotowanie danych do modelu
        X_train = Wynik_all[['Rok', 'Miesiąc', 'X', 'Y']].values
        y_train = Wynik_all['Opady'].values

        # Przygotowanie danych do przewidywania
        pred_mies_l = str(list(calendar.mies_name).index(mies_d[pred_mies])).zfill(2)
        X_pred = df_baza[['X', 'Y']].copy()
        X_pred['Rok'] = pred_rok
        X_pred['Miesiąc'] = pred_mies_l
        X_pred = X_pred[['Rok', 'Miesiąc', 'X', 'Y']].values

        # Przewidywanie
        y_pred = predict_rainfall(X_train, y_train, X_pred)
        df_baza['Opady'] = y_pred

        # Wyświetlanie przewidywanych danych
        st.title('Przewidywane Opady')
        st.dataframe(df_baza, width=800, height=1200)

        max_p = df_baza['Opady'].max()
        min_p = df_baza['Opady'].min()

        st.write(f"Maksymalna przewidywana ilość opadów: {max_p}")
        st.write(f"Minimalna przewidywana ilość opadów: {min_p}")

        # Rysowanie mapy 
        fig_pred, ax_pred = plot_wynik(shp, df_baza, f'Przewidywane opady {pred_mies} {pred_rok}')
        st.pyplot(fig_pred)
    else:
        st.error("Nie udało się załadować danych do przewidywania.")
