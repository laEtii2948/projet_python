import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt


def lire_plant_generation_data_groupe_3(filename : str) -> pd.DataFrame | None :
    """Fonction qui va lire le fichier csv plant generation data groupe 3 et renvoyer un dataframe.
    Si on ne trouve pas le fichier, on lève l'exception FileNotFoundError
    :param filename: le nom du fichier que l'on souhaite lire 
    :return: deux possibilité, on try de lire le fichier en passant le nom du fichier. Si le fichier n'est pas trouvé on lève l'exception"""
    
    try : 
        data_plant_generation = pd.read_csv(filename)
        st.success(f"Fichier trouvé : {filename}")
        return data_plant_generation 
    except Exception as e : 
        st.error(f"Fichier introuvable : {filename}")
        return None
    

def lire_plant_weather_data_group_3(filename : str) -> pd.DataFrame | None : 
    
    try :
        data_plant_weather = pd.read_csv(filename)
        st.success(f"Fichier trouvé : {filename}")
        return data_plant_weather
    except Exception as e : 
        st.error(f"Fichier introuvable : {filename}")
        return None
    

def conversion_en_date_time(df : pd.DataFrame) -> pd.DataFrame : 
    
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df


def tracer_production(df : pd.DataFrame, title : str = "Evolution temporelle de la production", xlabel : str ="Date", ylabel : str = "Puissance en courant continu (DC)") -> None :
    plt.figure(figsize=(12, 6)) 
    plt.plot(df["date_time"], df["dc_power"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def tracer_meteo_temp_ambiante(df: pd.DataFrame, title : str = "Evolution temporelle des données de températures ambiantes", xlabel : str = "Date", ylabel : str = "Température ambiante en °C") -> None :
    plt.figure(figsize=(12, 6)) 
    plt.plot(df["date_time"], df["ambient_temperature"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def tracer_meteo_module_temp(df: pd.DataFrame, title : str = "Evolution temporelle des données de températures des panneaux photovoltaïques", xlabel : str = "Date", ylabel : str = "Température des panneaux en °C") -> None :
    plt.figure(figsize=(12, 6)) 
    plt.plot(df["date_time"], df["module_temperature"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def tracer_meteo_irradiation(df: pd.DataFrame, title : str = "Evolution temporelle des données de températures des irradiations", xlabel : str = "Date", ylabel : str = "Irradiation solaire (en W/m²)") -> None :
    plt.figure(figsize=(12, 6)) 
    plt.plot(df["date_time"], df["irradiation"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)



    
    
        


st.set_page_config(
page_title="Projet python - Master MIMO - 2025",
page_icon=":python:",
layout="wide",
initial_sidebar_state="expanded",
)

st.header("Projet Python - Prédiction de la production solaire")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Présentation du projet","Exploration des données", "Préparation des données", "Importance des features et étude de corrélations entre les variables", "Modélisation", "Visualisation des résultats"])
with tab1:
    st.subheader("Présentation du projet")
    st.write("hello, coucoucdjfjdopjvopjfovfdvf")
with tab2:
    st.subheader("Exploration des données")
    st.markdown("Exploration des données: lire les données plant_generation_data_groupe_<1, 2,3>.csv et plant_weather_data_groupe_<1, 2, 3>.csv")
    lire_les_donnees = st.button("Lire les données")
    
    if lire_les_donnees : 
        df_plant_generation = lire_plant_generation_data_groupe_3("plant_generation_data_groupe_3.csv")
        df_weather_data = lire_plant_weather_data_group_3("plant_weather_data_groupe_3.csv")
        
    st.markdown("1. Quels sont les types de variables disponibles ?")
    
    st.write("Voici la liste des variables de production avec leurs types")
    st.dataframe(df_plant_generation.dtypes)

    st.write("Voici la liste des variables méteo avec leurs types")
    st.dataframe(df_weather_data.dtypes)
    
    st.markdown("2. Quelles sont les plages temporelles couvertes ?")
    conversion_en_date_time(df_plant_generation)
    conversion_en_date_time(df_weather_data)
    
    
    st.markdown("Données de production :")
    st.markdown(f"Données de production la plus ancienne : {df_plant_generation["date_time"].min()}")
    st.markdown(f"Données de production la plus récente : {df_plant_generation["date_time"].max()}")
    
    
    st.markdown("Données météo :")
    st.markdown(f"Données de production la plus ancienne : {df_weather_data["date_time"].min()}")
    st.markdown(f"Données de production la plus récente : {df_weather_data["date_time"].max()}")
    
    st.markdown("3. Visualiser l'évolution temporelle de la production d'énergie solaire et des données météorologiques")
    tracer_production(df_plant_generation)
    st.pyplot(plt)
    tracer_meteo_temp_ambiante(df_weather_data)
    st.pyplot(plt)
    tracer_meteo_module_temp(df_weather_data)
    st.pyplot(plt)
    tracer_meteo_irradiation(df_weather_data)
    st.pyplot(plt)
    
    
    
    
    

    
with tab3:
    st.subheader("Préparation des données")
with tab4:
    st.subheader("Importance des features et étude de corrélations entre les variables")
with tab5 : 
    st.subheader("Modélisation")
with tab6 : 
     st.subheader("Visualisation des résultats")
   
    

