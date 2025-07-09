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
    

def a_des_valeur_manquante(df : pd.DataFrame) -> pd.DataFrame : 
    nb_valeur_manquante_colonne = df.isnull().sum()
    
    total_valeur_manquante = nb_valeur_manquante_colonne.sum()
    
    if total_valeur_manquante == 0 : 
        st.success("Aucunes valeurs manquantes !")
    else : 
        st.warning("Des valeurs manquantes ont été détectées !")
        return nb_valeur_manquante_colonne
    

def fusionner_les_dataframe(df_plant_generation : pd.DataFrame, df_weather_data : pd.DataFrame) -> pd.DataFrame : 
    
    dataframe_fusionne = pd.merge(df_plant_generation,df_weather_data,on=["date_time", "plant_id"],how="inner")
    st.success("Dataframes de production et météorologiques fusionnées avec succès")
    return dataframe_fusionne


def create_features(df : pd.DataFrame) -> pd.DataFrame : 
    df["Jour de la semaine"] = df["date_time"].dt.day_of_week
    df["Jour"] = df["date_time"].dt.day_name()
    df["Heure"] = df["date_time"].dt.hour
    df["is_day"] = df["Heure"].between(6, 17).astype(int)
    return df
    
    
    




    
    
        


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
    st.markdown("1. Y'a-t-il des valeurs manquantes ?")
    st.markdown("Vérification pour les données de productions : ")
    a_des_valeur_manquante(df_plant_generation)
    st.markdown("Vérification pour les données météorologiques : ")
    a_des_valeur_manquante(df_weather_data)
    
    st.markdown("2. Fusionner les jeux de données météo et production")
    df_fusion = fusionner_les_dataframe(df_plant_generation,df_weather_data)
    st.dataframe(df_fusion)
    
    st.markdown("3. Créer des variables calendaires à partir de la variable date_time :")
    df_nouvelles_features = create_features(df_fusion)
    st.dataframe(df_nouvelles_features)
    
    
    
    
with tab4:
    st.subheader("Importance des features et étude de corrélations entre les variables")
with tab5 : 
    st.subheader("Modélisation")
with tab6 : 
     st.subheader("Visualisation des résultats")
   
    

