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
        st.error("Nie udało się pobrać danych z podanego URL: " + url)
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
        st.error("Nie udało się pobrać danych stacji z podanego URL: " + url)
        return None
    data = response.content.decode('windows-1250')
    df = pd.read_csv(StringIO(data), delimiter=',', header=None)
    
    df.columns = ['X', 'Y', 'Stacja']
    df['X'] = df['X'].astype(float)
    df['Y'] = df['Y'].astype(float)
    
    return df

# Function to calculate sum of rainfall
def suma_opadow(tabela):
    if tabela is None:
        return None
    tabela['Opady'] = pd.to_numeric(tabela['Opady'], errors='coerce')
    df_suma = tabela.groupby('Stacja')['Opady'].sum().reset_index()
    df_suma['Opady'] = df_suma['Opady'].astype(float)
    return df_suma

def plot_wynik(path_shapefile, Wynik, title):
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
    cbar = ax.contourf(X_siatka, Y_siatka, Z_siatka, cmap='YlGnBu', levels=np.arange(0, 450, 10), extend='min')
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
year = st.selectbox("Wybierz rok", [2010,2011,2012,2013,2014,2015,2016,2017,2018, 2019, 2020, 2021, 2022])
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
st.dataframe(df_baza)

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
    fig, ax = plot_wynik(path_shapefile, Wynik, f'Opady {month} {year}')
    st.pyplot(fig)
else:
    st.error("Nie udało się załadować danych.")

# Prediction Section
st.title("Przewidywanie Opadów")

# Wybór zakresu dat do przewidywania
start_year = st.selectbox("Wybierz rok początkowy", [2010,2011,2012,2013,2014,2015,2016,2017,2018, 2019, 2020, 2021])
end_year = st.selectbox("Wybierz rok końcowy", [2011,2012,2013,2014,2015,2016,2017,2018, 2019, 2020, 2021, 2022])
start_month = st.selectbox("Wybierz miesiąc początkowy", list(months_dict.keys()))
end_month = st.selectbox("Wybierz miesiąc końcowy", list(months_dict.keys()))

# Wybór docelowej daty do przewidywania
pred_year = st.selectbox("Wybierz rok do przewidywania", [2024, 2025, 2026])
pred_month = st.selectbox("Wybierz miesiąc do przewidywania", list(months_dict.keys()))

if st.button("Przewiduj"):
    # Zbieranie danych z wybranego zakresu
    all_data = []
    for yr in range(start_year, end_year + 1):
        for mnth in months_dict.keys():
            month_number = str(list(calendar.month_name).index(months_dict[mnth])).zfill(2)
            path_csv = f'https://raw.githubusercontent.com/Ladonean/FigDetect/main/o_d_{month_number}_{yr}.csv'
            df_temp = wczytaj_csv(path_csv)
            if df_temp is not None:
                df_temp['Rok'] = yr
                df_temp['Miesiąc'] = month_number
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
        pred_month_number = str(list(calendar.month_name).index(months_dict[pred_month])).zfill(2)
        X_pred = df_baza[['X', 'Y']].copy()
        X_pred['Rok'] = pred_year
        X_pred['Miesiąc'] = pred_month_number
        X_pred = X_pred[['Rok', 'Miesiąc', 'X', 'Y']].values

        # Przewidywanie
        y_pred = predict_rainfall(X_train, y_train, X_pred)
        df_baza['Opady'] = y_pred

        # Wyświetlanie przewidywanych danych
        st.title('Przewidywane Opady')
        st.dataframe(df_baza, width=800, height=1200)

        max_pred_value = df_baza['Opady'].max()
        min_pred_value = df_baza['Opady'].min()

        st.write(f"Maksymalna przewidywana ilość opadów: {max_pred_value}")
        st.write(f"Minimalna przewidywana ilość opadów: {min_pred_value}")

        # Rysowanie mapy przewidywań
        fig_pred, ax_pred = plot_wynik(path_shapefile, df_baza, f'Przewidywane opady {pred_month} {pred_year}')
        st.pyplot(fig_pred)
    else:
        st.error("Nie udało się załadować danych do przewidywania.")
