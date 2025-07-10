import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


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
    df["Jour"] = df["date_time"].dt.day
    df["Mois"] = df["date_time"].dt.month
    df["Heure"] = df["date_time"].dt.hour
    df["is_day"] = df["Heure"].between(6, 17).astype(int)
    return df


def separer_variables(df : pd.DataFrame) -> pd.DataFrame :
     x = df.drop(["date_time", "plant_id", "dc_power", "ac_power"],axis=1)
     y = df["dc_power"]
     st.success("Les données ont bien été séparées. D'un côté x les variables explicatives et y la variable cible ici dc_power ")
     return x,y
 
 
def tracer_feature_importance(x, indices, importances, feature_names, title : str = "Importances des caractéristiques") -> None :
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(x.shape[1]), importances[indices], align="center")
    plt.xticks(range(x.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, x.shape[1]])
    plt.show()


def afficher_le_top5(): 
    top_features = feature_names[indices][:5] 
    return top_features

def determiner_correlation(df : pd.DataFrame, top5): 
     df_correlation = df[top5].corr()
     return df_correlation
 
def tracer_heatmap_correlation(corr_variable, title="Matrice de corrélation")-> None :
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_variable, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    

def supprimer_module_temp(df : pd.DataFrame) -> pd.DataFrame :
     st.info("Colonne module_temperature supprimée car corrélée à irradiation")
     return df.drop("module_temperature", axis=1)

    
    




    
    
        


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
    st.markdown("1. Utiliser un modèle RandomForestRegressor pour évaluer l'importance des variables :")
    st.markdown("1. Séparer les données en X (variables explicatives) et y (variable cible)")
    x,y = separer_variables(df_nouvelles_features)
    st.markdown("2. Entraîner le modèle sur les données")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    st.spinner("Entraînement du modèle sur les données...")
    model.fit(x,y)
    st.success("Données entrainées avec succès !")
    st.markdown("3. Évaluer l'importance des variables avec model.feature_importances_")
    importances = model.feature_importances_
    feature_names = x.columns
    indices = np.argsort(importances)[::-1]
    tracer_feature_importance(x = x, indices = indices, importances = importances, feature_names = feature_names,title = "Importance des caractéristiques")
    st.pyplot(plt)
    st.markdown("2. Conserver uniquement les 5 variables les plus importantes pour l'entraînement dumodèle (celles avec la plus grande importance). Quelle est la variable la plus importante ?")
    top5 = afficher_le_top5()
    st.markdown(top5)
    st.markdown("La variable la plus importante est la variable irradiation")
    st.markdown("3. Y'a-t-il des variables corrélées entre elles dans les données sélectionnées ?")
    st.markdown("1. Visualiser la matrice de corrélation entre les variables (on utilisera df.corr() et une heatmap)")
    df_correlation = determiner_correlation(df_nouvelles_features, top5)
    st.dataframe(df_correlation)
    st.markdown("Visualisation des corrélations à travers une heatmap :")
    tracer_heatmap_correlation(df_correlation,title="Corrélations – Top-5 variables")
    st.pyplot(plt)
    st.markdown("2. Identifier les variables corrélées entre elles (corrélation > 0.9)")
    mask = (df_correlation > 0.9) & (df_correlation < 1)
    variable_correlee = df_correlation.where(mask)
    st.dataframe(variable_correlee)
    st.markdown("Les deux variables corrélées entre elles sont donc irradiation et module temperature")
    st.markdown("3. Supprimer une des deux variables corrélées (la moins importante) de l'analyse")
    df_final = supprimer_module_temp(df_correlation)
    st.dataframe(df_final)
    st.markdown("4. Créer une fonction create_final_dataset(df) qui prend en entrée un DataFrame et renvoie un DataFrame avec les variables sélectionnées pour l'entraînement du modèle.")
    
    
    
    
    
    
    
    
    
with tab5 : 
    st.subheader("Modélisation")
with tab6 : 
     st.subheader("Visualisation des résultats")
   
    

