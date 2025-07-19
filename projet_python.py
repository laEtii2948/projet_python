# --------------------------------------------------------------------------------------------------------------------
#                                                      IMPORTS
# --------------------------------------------------------------------------------------------------------------------

import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------------------------------------------
#                                                    PARTIE FONCTIONS 
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------FONCTION LIRE LE JEU DE DONNEES PLANT GENERATION------------------------------------
def lire_plant_generation_data_groupe_3(filename : str) -> pd.DataFrame | None :
    """Fonction qui va lire le fichier csv plant generation data groupe 3 et renvoyer un dataframe.
    Si on ne trouve pas le fichier, on lève l'exception FileNotFoundError
    :param filename: le nom du fichier que l'on souhaite lire 
    :return pd.Dataframe: Si le fichier est trouvé, la fonction renvoie un dataframe du csv 
    :return None: Si le fichier n'est pas trouvé, la fonction ne retourne rien à part un message d'erreur"""
    
    try : 
        data_plant_generation = pd.read_csv(filename)
        st.success(f"Fichier trouvé : {filename}")
        return data_plant_generation 
    except Exception as e : 
        st.error(f"Fichier introuvable : {filename}")
        return None


# --------------------------------FONCTION LIRE LE JEU DE DONNEES TARGET GROUPE--------------------------------------
def lire_target_groupe_3(filename : str) -> pd.DataFrame | None : 
    """Fonction qui va lire le fichier csv target groupe et renvoyer un dataframe. 
    Si on ne trouve pas le fichier, on lève une exception FileNotFoundError. 
    :param filename: le nom du fichier que l'on souhaite lire 
    :return pd.Dataframe: Si le fichier est trouvé, la fonction renvoie une dataframe du csv 
    :return None: Si le fichier n'est pas trouvé, la fonction ne retourne rien à part un message d'erreur."""
    
    try : 
        target_groupe_3 = pd.read_csv(filename)
        st.success(f"Fichier trouvé : {filename}")
        return target_groupe_3
    except Exception as e : 
        st.error(f"Fichier introuvable : {filename}")
        return None
    

# --------------------------------FONCTION LIRE LE JEU DE DONNEES WEATHER FORECAST GROUPE--------------------------------
def lire_plant_weather_forecast_groupe_3(filename : str) -> pd.DataFrame | None : 
    """Fonction qui va lire le fichier csv weather forecast groupe et renvoyer un dataframe. 
    Si on ne trouve pas le fichier, on lève une exception FileNotFoundError. 
    :param filename: le nom du fichier que l'on souhaite lire 
    :return pd.Dataframe: Si le fichier est trouvé, la fonction renvoie un dataframe du csv 
    :return None: Si le fichier n'est pas trouvé, la fonction ne retourne rien à part un message d'erreur"""
    
    try : 
        plant_weather_forecast_groupe_3 = pd.read_csv(filename)
        st.success(f"Fichier trouvé : {filename}")
        return plant_weather_forecast_groupe_3
    except Exception as e : 
        st.error(f"Fichier introuvable : {filename}")
        return None
    

# --------------------------------FONCTION LIRE LE JEU DE DONNEES WEATHER DATA GROUP-----------------------------------
def lire_plant_weather_data_group_3(filename : str) -> pd.DataFrame | None : 
    """Fonction qui va lire le fichier csv weather data group et renvoyer un dataframe. 
    Si on ne trouve pas le fichier, on lève une exception FileNotFoundError. 
    :param filename: le nom du fichier que l'on souhaite lire 
    :return pd.Dataframe: Si le fichier est trouvé, la fonction renvoie un dataframe du csv 
    :return None: Si le fichier n'est pas trouvé, la fonction ne retourne rien à part un message d'erreur"""
    
    try :
        data_plant_weather = pd.read_csv(filename)
        st.success(f"Fichier trouvé : {filename}")
        return data_plant_weather
    except Exception as e : 
        st.error(f"Fichier introuvable : {filename}")
        return None
    
# --------------------------------FONCTION DE CONVERSION EN TYPE DATE TIME-------------------------------------------
def conversion_en_date_time(df : pd.DataFrame) -> pd.DataFrame : 
    """Convertit la colonne date time dd'un dataframe en type datetime de la librairie pandas, ce qui va nous permettre 
    par la suite d'appliquer des méthodes de sur ce type pour travailler sur les aspects calendaires de cette colonne. 
    La colonne est remplacée directement dans le dataframe fourni. 
    :param df: dataframe contenant une colonne date_time au format autre que datetime
    :return pd.Dataframe: Retourne le même dataframe fourni avec la colonne date_time désormais en type datetime de pandas."""
    
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df


# --------------------------------FONCTION TRACER LA COURBE TEMPORELLE DE PRODUCTION----------------------------------
def tracer_production(df : pd.DataFrame, title : str = "Evolution temporelle de la production", xlabel : str ="Date au format Mois-Jour-Heure", ylabel : str = "Puissance en courant continu (DC)") -> None :
    """Fonction qui trace l'évolution temporelle de la puissance en courant continu (DC) à partir des données de 
    production solaire. La fonction créée le graphique matplotlib puis affiche la courbe temporelle. 
    :param df: dataframe contenant obligatoirement les colonnes date_time et dc_power pour les besoins de la visualisation
    :param title: Titre du graphique. Par défaut, c'est celui indiqué en paramètre
    :param xlabel: Libellé de l'axe x du graphique. Par défaut, c'est celui indiqué en paramètre
    :param ylabel: libellé de l'axe y. Par défaut, c'est celui indiqué en paramètre. 
    :return None: La fonction ne renvoie rien, elle génère le graphique."""
    
    plt.figure(figsize=(12, 6)) 
    plt.plot(df["date_time"], df["dc_power"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# -----------------------------FONCTION TRACER LA COURBE TEMPORELLE DES DONNEES METEOROLOGIQUES---------------------------- 
def tracer_meteo_temp_ambiante(df: pd.DataFrame, title : str = "Evolution temporelle des données de températures ambiantes", xlabel : str = "Date au format Mois-Jour-Heure", ylabel : str = "Température ambiante en °C") -> None :
    """Cette fonction trace la courbe temporelle des données météorologiques. Ici on commence par les données de températures
    ambiantes. 
    :param df: dataframe avec les données météorologiques. Le dataframe doit au moins contenir date_time et ambient_temperature
    :param title: Titre du graphique. Par défaut, c'est celui indiqué en paramètre 
    :param xlabel: Libellé de l'axe x du graphique. Par défaut, c'est celui indiqué en paramètre
    :param ylabel: Libellé de l'axe y du graphique. Par défaut, c'est celui indiqué en paramètre. """
    
    plt.figure(figsize=(12, 6)) 
    plt.plot(df["date_time"], df["ambient_temperature"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# -----------------------------FONCTION TRACER LA COURBE TEMPORELLE DES DONNEES METEOROLOGIQUES----------------------------   
def tracer_meteo_module_temp(df: pd.DataFrame, title : str = "Evolution temporelle des données de températures des panneaux photovoltaïques", xlabel : str = "Date au format Mois-Jour-Heure", ylabel : str = "Température des panneaux en °C") -> None :
    """Cette fonction trace la courbe temporelle des données météorologiques. Ici, les données des températures des panneaux photovoltaïques
    :param df: dataframe avec les données météorologiques. Le dataframe doit au moins contenir date_time et module_temperature
    :param title: Titre du graphique. Par défaut, c'est celui indiqué en paramètre 
    :param xlabel: Libellé de l'axe x du graphique. Par défaut, c'est celui indiqué en paramètre
    :param ylabel: Libellé de l'axe y du graphique. Par défaut, c'est celui indiqué en paramètre. """
    
    plt.figure(figsize=(12, 6)) 
    plt.plot(df["date_time"], df["module_temperature"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# -----------------------------FONCTION TRACER LA COURBE TEMPORELLE DES DONNEES METEOROLOGIQUES---------------------------- 
def tracer_meteo_irradiation(df: pd.DataFrame, title : str = "Evolution temporelle des données de températures des irradiations", xlabel : str = "Date au format Mois-Jour-Heure", ylabel : str = "Irradiation solaire (en W/m²)") -> None :
    """Cette fonction trace la courbe temporelle des données météorologiques. Ici les données des températures des irradiations solaires.
    :param df: dataframe avec les données météorologiques. Le dataframe doit au moins contenir date_time et irradiation 
    :param title: Titre du graphique. Par défaut, c'est celui indiqué en paramètre 
    :param xlabel: Libellé de l'axe x du graphique. Par défaut, c'est celui indiqué en paramètre 
    :param ylabel: Libellé de l'axe y du graphique. Par défaut, c'est celui indiqué en paramètre. """
    
    plt.figure(figsize=(12, 6)) 
    plt.plot(df["date_time"], df["irradiation"])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


# -----------------------------FONCTION DE VERIFICATION DES VALEURS MANQUANTES--------------------------------------------
def a_des_valeur_manquante(df : pd.DataFrame) -> pd.Series | None : 
    """Vérifie la présence de valeurs manquantes dans un dataframeet affiche un message sur streamlit approprié en fonction 
    de la situation. La fonction calcule pour chaque colonnes le nombre de valeurs NaN
    :param df: le dataframe dont on veut faire la vérification des valeurs lanquantes. 
    :return pd.Series: Si des valeurs manquantes existent, on retourne le nombre de valeurs manquantes, ainsi qu'un message
    de warning streamlit
    :return None: Si le dataframe est complet, c'est-à-dire qu'il n'existe pas de NaN, la fonction ne renvoie rienet indique
    par un message qu'il n'y a aucune présence de valeurs manquantes dans le jeu de données. """

    nb_valeur_manquante_colonne = df.isnull().sum()
    
    total_valeur_manquante = nb_valeur_manquante_colonne.sum()
    
    if total_valeur_manquante == 0 : 
        st.success("Aucunes valeurs manquantes !")
    else : 
        st.warning("Des valeurs manquantes ont été détectées !")
        return nb_valeur_manquante_colonne
    

# -----------------------------FONCTION DE FUSION DE DEUX DATAFRAMES----------------------------------------------
def fusionner_les_dataframe(df_plant_generation : pd.DataFrame, df_weather_data : pd.DataFrame) -> pd.DataFrame : 
    """Fonction qui fusionne les données de production électrique et les données météorologiques sur les clés 
    communes date_timeet plant_id. Un pd.merge est effectué avec une jointure garantissant que seules les lignes 
    présentes dans les deux jeux de donénes sont conservées. Une notificationstreamlit confirme la réussite de l'opération. 
    :param df_plant_generation: dataframe contenant au moins date_time et plant_id, c'est le jeu de données de production électrique
    :param df_weather_data: dataframe contenant au moins date_time et plant_id, c'est le jeu de données météorologiques. 
    :return pd.Dataframe: Retourne un dataframe fusionné, incluant toutes les colonnes d’origine provenant des deux sources pour les lignes correspondantes."""
    
    dataframe_fusionne = pd.merge(df_plant_generation,df_weather_data,on=["date_time", "plant_id"],how="inner")
    st.success("Dataframes de production et météorologiques fusionnées avec succès")
    return dataframe_fusionne


# -----------------------------FONCTION DE FUSION DE DEUX DATAFRAMES----------------------------------------------
def create_features(df : pd.DataFrame) -> pd.DataFrame :
    """Fonction qui ajoute des features calendaires dérivées de la colonne date_time et renvoie le dataframe enrichi. 
    :param df: dataframe à enrichir contenant au moins une colonne date_time au type datetime de pandas. 
    :return pd.datafrme: Retourne le mêm dataframe enrichi de cinq colonnes (jour de la semaine, jour, mois, heure, is_day)""" 
    
    df["Jour de la semaine"] = df["date_time"].dt.day_of_week
    df["Jour"] = df["date_time"].dt.day
    df["Mois"] = df["date_time"].dt.month
    df["Heure"] = df["date_time"].dt.hour
    df["is_day"] = df["Heure"].between(6, 17).astype(int)
    return df


# -----------------------------FONCTION DE SEPARATION DES VARIABLES ----------------------------------------------
def separer_variables(df : pd.DataFrame) -> pd.DataFrame | pd.Series :
     """ Fonction qui sépare un dataframe (production + météo) en : 
     x : les variables explicatives 
     y : la variable cible (dc_power), celle que l'on cherche à expliquer. 
     La fonction supprime d'abord les colonnes non explicatives (date_time, `plant_id`, `dc_power`, `ac_power`) pour construire x,
     puis extrait la colonne `dc_power` comme **y**. Elle affiche ensuite via Streamlit, un message de confirmation.
     :param df: DataFrame combinant données de production et météo
     :return x: dataframe des variables explicatives 
     :return y: serie panda correspondant à dc_power"""
     
     x = df.drop(["date_time", "plant_id", "dc_power", "ac_power"],axis=1)
     y = df["dc_power"]
     st.success("Les données ont bien été séparées. D'un côté x les variables explicatives et y la variable cible ici dc_power ")
     return x,y
 
# -----------------------------FONCTION TRACER LES FEATURES IMPORTANCES ---------------------------------------------
def tracer_feature_importance(x , indices , importances, feature_names, title : str = "Importances des caractéristiques") -> None :
    """Fonction qui affiche un graphique en bar des importances de variables produites par un modèle. 
    Le graphique affiche dans l'ordre décroissant d'importance la contribution de chaque *feature* à la prédiction.
    :param x: Jeu de données d'origine utilisé pour connaître le nombre de variables. 
    :param indices: Indices triés des variables déterminant l'ordre d'affichage des variables. 
    :param importances: Valeurs numériques représentant l'importance de chaque variables
    :param features_names: nom des variables dans le même ordre que la variable importances. 
    :param title: Titre du graphique. Par défaut, celle indiqué en paramètre. """
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(x.shape[1]), importances[indices], align="center")
    plt.xticks(range(x.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, x.shape[1]])
    plt.show()

# -----------------------------FONCTION AFFICHAGE DU TOP 5 ------------------------------------------------------
def afficher_le_top5() -> np.ndarray : 
    """Fonction qui renvoie les cinq variables les plus importantes selon l'ordre de tri contenu dans indices. Les
    cinq premiers éléments de features_names sont extraits puis renvoyés. 
    return np.ndarray : Les noms des cinq variables jugées les plus importantes. """
    
    top_features = feature_names[indices][:5] 
    return top_features


# -----------------------------FONCTION DETERMINER CORRELATION----------------------------------------------------
def determiner_correlation(df : pd.DataFrame, top5) -> pd.DataFrame: 
     """Calcule la matrice de correlation des cinq variables les plus importantes et la renvoie sous forme de dataframe. 
     :param df: dataframe contenant les données du top5 
     :param top5: Nom des cinq variables pour lesquelles la corrélation doit être calculée. """
     
     df_correlation = df[top5].corr()
     return df_correlation
 

# -----------------------------FONCTION TRACER HEATMAP-----------------------------------------------------------
def tracer_heatmap_correlation(corr_variable : pd.DataFrame, title="Matrice de corrélation")-> None :
    """Fonction qui affiche une heatmap avec la matrice de corrélation passée en argument.
    :param corr_variable: matrice de coefficients de corrélation. 
    :param title: Titre du graphique. Par défaut, ce qui est indiqué en paramètre. 
    :return None: La fonction ne renvoie rien. Elle affiche juste la heatmap. """
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_variable, annot=True, fmt=".2f", cmap="coolwarm", square=True)


# -----------------------------FONCTION SUPPRIMER MODULE TEMP------------------------------------------------------
def supprimer_module_temp(df : pd.DataFrame) -> pd.DataFrame :
     """Fonction qui supprime la colonne module_temperature d'un dataframe car elle est fortement corrélée à la variable irradiation. 
     1. La fonction affiche un message informatif sur streamlit pour expliquer la suppression. 
     2. Retourne le dataframe sans cette colonne. 
     :param df: dataframe contenant le jeu de données et surtout contenant la colonne module_temperature 
     :return pd.Dataframe: Retourne ce même dataframe sans la colonne module_temperature. """
     
     st.info("Colonne module_temperature supprimée car corrélée à irradiation")
     return df.drop("module_temperature", axis=1)


# -----------------------------FONCTION CREATION DATASET FINAL------------------------------------------------------
def create_final_dataset(df : pd.DataFrame) -> pd.DataFrame : 
    """Fonction qui sélectionne les variables explicatives jugées déterminantes pour l'entraînement des modèles et renvoie
    un dataframe final. La fonction renvoie une copie pour laisser intact le dataframe d'origine. 
    :param df: Jeu de données complet contenant les quatres variables explicatives. 
    :return pd.Dataframe: Nouveau dataframe final limité aux colonnes sélectionnées et dans l'ordre spécifique. """
    
    colonnes_selectionees = ["irradiation", "ambient_temperature", "Heure", "Jour"]
    return df[colonnes_selectionees].copy()


# -----------------------------FONCTION DE SEPARATION DES VARIABLES X ET Y--------------------------------------------
def separer_X_y(df : pd.DataFrame) -> pd.DataFrame | pd.Series : 
    """Fonction qui sépare le jeu de données complet en : 
    X : variables explicatives, limitées aux quatres colonnes retenues par la fonction create final dataset 
    y : variable cible, dc_power que l'on cherche à prédire. 
    :param df: dataframe final contenant toutes les données nécessaire pour entrainer le modèle et effectuer les prédictions
    :return X: x est le dataframe à quatre colonnes avec les variables explicatives. 
    :return y: y est une série pandas contenant la variable cible dc_power"""
    
    X = create_final_dataset(df)
    y = df["dc_power"]
    st.info("Les données ont été séparées entre variables explicatives (X) et variable cible (y)")
    return X,y 


# -----------------------------FONCTION DE SEPARATION ENTRE DONNEES ENTRAINEMENT ET DONNEES DE TEST----------------------
def split_train_test(X : pd.DataFrame, y : pd.Series, test_size : float = 0.2, random_state : int = 42) -> pd.DataFrame | pd.Series : 
    """"Fonction qui scinde le jeu de données en jeu de données d'entrainement à auteur de 80% et jeu de données de test à hauteur de 20% 
    Jeu d'entrainement : c'est le jeu de données utilisé pour l'apprentissage du modèle 
    Jeu de test : c'est le jeu de données pour l'évaluation finale de notre modèle. 
    Un message informatif est affiché dans streamlit pour rappeler le découpage retenu. 
    :param X: Variables explicatives à séparer
    :param y: Variable cible à séparer
    :param test_size: Proportion du jeu de test. Ici, c'est 20% (0.2)
    :param random_state: Génération aléatoire.
    :return X: dataframe contenant les observations d'entrainement
    :return y: retourne une serie pandes pour les cibles d'entrainement"""
    
    st.info("Les données ont été découpée en 80% de données d'entrainement et 20% de données de tests")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# -----------------------------FONCTION EVALUATION MODELE-------------------------------------------------------------
def evaluer_modele(y_test, y_pred) -> pd.DataFrame :
    """Fonction qui calcule quatre métriques classiques d'évaluation d'un modèle et les renvoie dans un dataframe. 
    mae : mean absolute error - Ecart entre la valeur réelle et prédite
    mse : mean squared error - total des erreurs : amplifie les ecarts 
    rmse : root mean squared error - mesure typique de l'erreur mais avec une sensibilité de la MSE 
    R² : Mesure de l'efficacité du modèle. 
    :param y_test: valeurs réelles de la variable cible sur le jeu de test 
    :param y_pred: valeurs prédites par le modèle sur ce même jeu """ 
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return pd.DataFrame({"MAE": [mae], "MSE": [mse], "RMSE": [rmse], "R²": [r2]})


# -----------------------------FONCTION TRACER GRAPHIQUE PREDICTION -------------------------------------------------------------
def tracer_graphique_prediction(y_test, y_pred_prev, title : str = "Production réelle VS production prédite", xlabel : str = "Puissance DC réelle", ylabel : str = "Puissance DC prédite") -> None : 
    """Fonction qui affiche un graphique en nuage de point des valeurs de tests et les valeurs prédites pour évaluer 
    visuellement la précision du modèle. Un trait pointillé rouge représente une droite de référence. Plus les points sont proches de cette droite
    plus les prédictions sont exactes. 
    :param y_test: Valeurs observées de la variables cible sur le jeu de test
    :param y_pred_prev: les valeurs prédites 
    :param title: Titre du graphique. Par défaut, c'est celle qui est indiqué en paramètre. 
    :param xlabel: Libellé de l'axe x. Par défaut, c'est celle qui est indiqué en paramètre
    :param ylabel: Libellé de l'axe y. Par défaut, c'est celle qui est indiqué en paramètre.
    :return None: La fonction ne retourne rien. Elle trace simplement le graphique. """
    
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred_prev, color="tab:blue", s=60, label = "prédictions")
    plt.scatter(y_test,y_test,facecolors="none", edgecolors="tab:orange", marker="o", s=70, label="réel (référence)")
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="red")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()


#-----------------------------FONCTION SEPARER LES DONNEES EXPLICATIVES ET CIBLE ----------------------------------------------------
def separer_X_ac_y_ac(df : pd.DataFrame) -> pd.DataFrame | pd.Series : 
    """Sépare un dataframe en deux ensemble de variables explicatives et cibles. 
    :param df: dataframe sur lequel on va travailler pour ensuite séparer les variables explicatives et cibles
    :return X_ac: retourne X_ac qui est un dataframe
    :return y_ac: retourne y_ac qui est une série pandas. """
    features_ac = ["irradiation","ambient_temperature","Heure","Jour","dc_power_pred"] 
    X_ac = df_nettoyage[features_ac].copy()
    y_ac = df_nettoyage["ac_power"].copy()
    st.info("Les données ont été séparées entre variables explicatives (X) et variable cible (y)")
    return X_ac, y_ac


#-----------------------------FONCTION TRACER GRAPHIQUE PREDICTION------------------------------------------------------------------
def tracer_graphique_prediction_ac(y_test, y_pred_ac, title : str = "Production alternative réelle VS production alternative prédite", xlabel : str = "Puissance AC réelle", ylabel : str = "Puissance AC prédite") -> None : 
    """Fonction qui trace un graphique en nuage de point de la puissance en courant alternatif réelle et la puissance en courant alternative prédite. 
    La fonction affiche une droite de référence rouge et évalue visuellement la précision du modèle. 
    :param y_test: variable de test réelle 
    :param y_pred_ac: Les prédictions du courant alternatif
    :param title: Titre du graphique. Par défaut c'est celle qui est en paramètre
    :param xlabel: Libellé de l'axe x du graphique. Par défaut, c'est celle qui est en paramètre 
    :param ylabel: Libellé de l'axe y du graphique. Par défaut, c'est celle qui est en paramètre. 
    :return None: La fonction ne retourne rien. Elle affiche simplment le graphique. """
    
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred_ac, color="tab:blue", s=60,label="predictions")
    plt.scatter(y_test, y_test, facecolors="none", edgecolors="tab:orange", marker="o", s=70, label="réel (référence)")
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="red")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    


#-----------------------------FONCTION TRACER EVOLUTION PRODUCTION COURANT ALTERNATIF-----------------------------------------------------
def tracer_evolution_prod_ac(df : pd.DataFrame, title : str = "Évolution temporelle de la puissance AC (prédite)", xlabel : str = "Date au format Mois-Jour-Heure", ylabel : str = "Puissance AC prédite") -> None : 
    """Fonction qui trace la courbe temporelle de la puissance en courant alternatif prédite afin de visualiser son évolution au fil du temps. 
    :param df: Le dataframe utilisé pour les besoins de la fonction 
    :param title: Titre du graphique. Par défaut c'est celle qui est indiqué en paramètre. 
    :param xlabel: Libellé de l'axe x. Par défaut c'est celle qui est indiqué en paramètre. 
    :param ylabel: Libellé de l'axe y. Par défaut c'est celle qui est indiqué en paramètre. 
    :return None: La fonction ne retourne rien. Elle affiche simplement le graphique. """
    
    plt.figure(figsize=(12, 6))
    plt.plot(df["date_time"], df["ac_power_pred"], label="AC prédite")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    

#-----------------------------FONCTION TRACER EVOLUTION EFFICACITE ONDULATEUR----------------------------------------------------------
def tracer_evolution_efficacite_ondulateur(df : pd.DataFrame, title : str ="Evolution de l'efficacité ondulateur", xlabel : str= "Date au format Mois-Jour", ylabel : str = "Valeur (W ou %)") -> None : 
    """Fonction qui affiche sur ce même graphique l'évolution temporelle de la puissance en courant continu, puissance en courant alternatif et 
    de l'efficacité ondulateur. 
    :param df: le dataframe utilisé dans le cadre de ce graphique. 
    :param title: Titre du graphique. Par défaut, c'est celui indiqué en paramètre. 
    :param xlabel: Libellé de l'axe x du graphique. Par défaut, c'est celle indiqué en paramètre. 
    :param ylabel: Libellé de l'axe y du graphique. Par défaut, cest celle indiqué en paramètre.
    :return None: La fonction ne retourne rien. Elle affiche simplement le graphique."""
    
    plt.figure(figsize=(14, 6))
    plt.plot(df["date_time"], df["dc_power_pred"], label=" Puissance courant continu")
    plt.plot(df["date_time"], df["ac_power_pred"], label=" Puissance courant alternatif")
    plt.plot(df["date_time"], df["efficacité_pourcentage"] * 100, "--", label=" Efficacité ondulateur (en %)")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    

#-----------------------------FONCTION TRACER VISUALISATION RESIDUS----------------------------------------------------------------
def tracer_visualisation_residus(y_pred, residus, title : str = "Résidus en fonction des valeurs prédites", xlabel = "Valeurs prédites", ylabel : str ="Résidus (le résultat de y_test - y_pred)") -> None : 
    """Fonction qui trace un graphique scatter des résidus du modèle de régression linéaire en fonction des valeurs prédites.La droite de référence rouge 
    est la droite de référence sert de repère pour distinguer les sous-estimations et les surestimations. 
    :pram y_pred: valeurs prédites sur le modèles sur le jeu de test 
    :param residus: différences entre y_test et y_pred, en d'autre termes, les erreurs de prédicition. 
    :param title: Titre du graphique. Par défaut, c'est celle qui est indiqué dans les paramètres. 
    :param xlabel: Libellé de l'axe x du graphique. Par défaut, c'est celle qui est indiqué dans les paramètres. 
    :param ylabel: Libellé de l'axe y du graphique. Par défaut, c'est celle qui est indiqué dans les paramètres. 
    :return None: La fonction ne retourne rien. Elle affiche simplement le graphique. """
    
    plt.figure(figsize=(8,5))
    plt.scatter(y_pred, residus, alpha=0.5)
    plt.axhline(0, color="red", ls="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()



# --------------------------------------------------------------------------------------------------------------------
#                                          PARTIE AFFICHAGE STREAMLIT
# --------------------------------------------------------------------------------------------------------------------

# ------------------------------------------CONFIGURATION DE LA PAGE--------------------------------------------------
st.set_page_config(
page_title="Projet python - Master MIMO - 2025",
page_icon=":python:",
layout="wide",
initial_sidebar_state="expanded",
)

# ------------------------------------------PRESENTATION SOUS FORME D'ONGLETS------------------------------------------
st.header("Projet Python - Prédiction de la production solaire")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Exploration des données", "Préparation des données", "Importance des features et étude de corrélations entre les variables", "Modélisation", "Visualisation des résultats", "Bonus", "Résidus"])

# --------------------------------------------------------------------------------------------------------------------
#                                          PARTIE EXPLORATIONS DES DONNEES
# --------------------------------------------------------------------------------------------------------------------

with tab1:
    st.subheader("Exploration des données")
    st.markdown("Exploration des données: lire les données plant_generation_data_groupe_<1, 2,3>.csv et plant_weather_data_groupe_<1, 2, 3>.csv")
    lire_les_donnees = st.button("Lire les données d'explorations")
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
    st.markdown("Ce graphique montre la puissance continue (en courant continu) délivrée par une installation, très probablement photovoltaïque, du 7 au 16 juin 2020. Chaque journée forme une arche : la production démarre brusquement au lever du soleil, atteint rapidement un plateau oscillant entre 15 000 et 20 000 W, puis retombe à zéro au coucher, traduisant le cycle jour‑nuit typique du solaire. ")
    tracer_meteo_temp_ambiante(df_weather_data)
    st.pyplot(plt)
    st.markdown("Le tracé représente la température ambiante, mesurée heure par heure du 7 au 16 juin 2020. On y voit la courbe du cycle jour‑nuit : chaque journée débute par une valeur minimale au petit matin (environ 23–24 °C), la température grimpe rapidement en matinée, atteint un maximum vers le milieu de l’après‑midi (jusqu’à 34–35 °C les 8, 9 et 10 juin) puis redescend progressivement après le coucher du soleil. À partir du 12 juin, les sommets quotidiens s’abaissent autour de 30–32 °C et les creux deviennent un peu plus frais (~22 °C). Globalement, le graphique illustre une alternance thermique journalière stable, avec une première moitié de période très chaude puis un léger rafraîchissement en seconde moitié.")
    tracer_meteo_module_temp(df_weather_data)
    st.pyplot(plt)
    st.markdown("Le graphique retrace la température de surface des panneaux photovoltaïques du 7 au 16 juin 2020 : chaque journée s’ouvre à une vingtaine de degrés juste après l’aube, la température grimpe rapidement sous l’effet du rayonnement solaire, culmine entre 50 °C et plus de 60 °C lors des journées très ensoleillées (8–10 juin), puis redescend brutalement dès que l’irradiance baisse, avant de revenir autour de 22–25 °C pendant la nuit. Les bosses et creux intrajournaliers reflètent donc à la fois l’alternance jour‑nuit et les fluctuations météorologiques.")
    tracer_meteo_irradiation(df_weather_data)
    st.pyplot(plt)
    st.markdown("Ce tracé représente l’irradiation solaire reçue au sol entre le 7 et le 16 juin 2020. Comme prévu, le signal est nul la nuit, puis grimpe brutalement après le lever du soleil, forme un plateau plus ou moins élevé en milieu de journée et retombe à zéro au coucher : c’est donc le cycle journalier de l’ensoleillement.")


# --------------------------------------------------------------------------------------------------------------------
#                                          PARTIE PREPARATION DES DONNEES
# --------------------------------------------------------------------------------------------------------------------  
   
with tab2:
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


# --------------------------------------------------------------------------------------------------------------------
#                                  PARTIE IMPORTANCE DES FEATURE ET ETUDE DE CORRELATION
# --------------------------------------------------------------------------------------------------------------------     
    
with tab3:
    st.subheader("Importance des features et étude de corrélations entre les variables")
    st.markdown("1. Utiliser un modèle RandomForestRegressor pour évaluer l'importance des variables :")
    
    st.markdown("1. Séparer les données en X (variables explicatives) et y (variable cible)")
    x,y = separer_variables(df_nouvelles_features)
    
    st.markdown("2. Entraîner le modèle sur les données")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x,y)
    st.success("Données entrainées avec succès !")
    
    st.markdown("3. Évaluer l'importance des variables avec model.feature_importances_")
    importances = model.feature_importances_
    feature_names = x.columns
    indices = np.argsort(importances)[::-1]
    tracer_feature_importance(x = x, indices = indices, importances = importances, feature_names = feature_names,title = "Importance des caractéristiques")
    st.pyplot(plt)
    st.markdown("Ce diagramme de l’importance des caractéristiques montre qu’une variable domine très nettement toutes les autres : l’irradiation solaire (la première barre, environ  90 % de l’importance totale) influence presque à elle seule le modèle de prédiction. Les facteurs secondaires comme les température ambiante et température des panneaux ne pèsent que quelques % chacun, tandis que les indicateurs calendaires (jour, semaine, mois, day of week) sont quasi négligeables. Autrement dit, pour estimer la production photovoltaïque, le modèle s’appuiera avant tout sur la quantité de rayonnement reçu, les conditions thermiques n’apportent qu’un ajustement mineur, et le calendrier n’a qu’un effet marginal.")
    
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
    st.markdown("La matrice de corrélation montre que l’irradiation solaire est très fortement corrélée à la température des modules (0,96) et, dans une moindre mesure, à la température ambiante (0,75). Plus le rayonnement est élevé, plus les panneaux montent en température.")
    
    st.markdown("2. Identifier les variables corrélées entre elles (corrélation > 0.9)")
    mask = (df_correlation > 0.9) & (df_correlation < 1)
    variable_correlee = df_correlation.where(mask)
    st.dataframe(variable_correlee)
    st.markdown("Les deux variables corrélées entre elles sont donc irradiation et module temperature")
    
    st.markdown("3. Supprimer une des deux variables corrélées (la moins importante) de l'analyse")
    df_nettoyage = supprimer_module_temp(df_nouvelles_features)
    st.dataframe(df_nettoyage)
    
    st.markdown("4. Créer une fonction create_final_dataset(df) qui prend en entrée un DataFrame et renvoie un DataFrame avec les variables sélectionnées pour l'entraînement du modèle.")
    df_selected = create_final_dataset(df_nettoyage)
    st.dataframe(df_selected.head())


# --------------------------------------------------------------------------------------------------------------------
#                                                   PARTIE MODELISATION
# -------------------------------------------------------------------------------------------------------------------- 
     
with tab4 : 
    st.subheader("Modélisation")
    st.markdown("1. Séparer les données en train/test (80% des données pour l'entraînement et 20% pour le test)")
    X, y = separer_X_y(df_nettoyage)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    st.markdown("2. Entraîner le modèle de régression linéaire (LinearRegression) sur les données d'entraînement")
    model = LinearRegression()
    model.fit(X_train, y_train)
    st.success("Les données ont bien été entrainées avec le modèle de régression linéaire !")
    
    st.markdown("3. Évaluer le modèle sur les données de test")
    y_pred = model.predict(X_test)
    df_score = evaluer_modele(y_test, y_pred)
    st.success("Modèle évalué avec succès !")
    
    st.markdown("4. Afficher les métriques de performance (ex: RMSE, MAE, R2)")
    st.dataframe(df_score)
    
    st.markdown("5. Entraîner un modèle de régression par forêt aléatoire (RandomForestRegressor) sur les données d'entraînement")
    model_forest = RandomForestRegressor(random_state=42)
    model_forest.fit(X_train, y_train)
    st.success("Les données ont bien été entrainées avec le modèle de forêt aléatoire !")
    
    st.markdown("6. Évaluer le modèle sur les données de test")
    y_pred_random_forest = model_forest.predict(X_test)
    df_score_forest = evaluer_modele(y_test, y_pred_random_forest)
    st.success("Modèle évalué avec succès !")
    
    st.markdown("7. Afficher les métriques de performance (ex: RMSE, MAE, R2)")
    st.dataframe(df_score_forest)
    
    st.markdown("8. Comparer les performances des deux modèles")
    df_comparer_modeles = pd.concat([df_score, df_score_forest])
    st.dataframe(df_comparer_modeles)
    st.markdown("Au vu des résultats obtenus, le modèle de forêt aléatoire semble donner de meilleurs résultats")


# --------------------------------------------------------------------------------------------------------------------
#                                      PARTIE VISUALISATION DES RESULTATS
# -------------------------------------------------------------------------------------------------------------------- 
    
with tab5 : 
     st.subheader("Visualisation des résultats")
     st.markdown("1. Lire les données de test: target_groupe_<1, 2, 3>.csv et plant_weather_forecast_groupe_<1, 2, 3>.csv")
     st.button("Lire les données")
     df_to_pred = lire_target_groupe_3("target_groupe_3.csv")
     df_weather_forecast = lire_plant_weather_forecast_groupe_3("plant_weather_forecast_groupe_3.csv")
     
     st.markdown("2. Fusionner les données de test avec les prévisions météo")
     df_to_pred = fusionner_les_dataframe(df_to_pred, df_weather_forecast)
     st.dataframe(df_to_pred)
     
     st.markdown("3. Appliquer les transformations nécessaires")
     conversion_en_date_time(df_to_pred)
     df_to_pred_features = create_features(df_to_pred)
     df_to_pred_selected = create_final_dataset(df_to_pred_features)
     st.dataframe(df_to_pred_selected.head())
     
     st.markdown("4. Appliquer le meilleur modèle (celui ayant la métrique R2 la plus élevée) sur les données de test")
     y_pred_prev = model_forest.predict(X_test)
     results = X_test.copy() 
     results["actual"] = y_test.values     
     results["prédit"] = y_pred_prev
     st.dataframe(results.head())
     tracer_graphique_prediction(y_test, y_pred_prev)
     st.pyplot(plt)
     st.markdown("Le nuage de points compare pour un ensemble d’observations la puissance DC réellement mesurée à celle estimée par le modèle. La diagonale rouge représente l’accord parfait (y = x). La majorité des points s’alignent assez près de cette diagonale, signe que le modèle reproduit globalement la production. ")


# --------------------------------------------------------------------------------------------------------------------
#                                                        PARTIE BONUS
# -------------------------------------------------------------------------------------------------------------------- 

with tab6 : 
    st.subheader("Bonus")
    st.markdown("6. Prédire la valeur de ac_power (la puissance en courant alternatif) à partir des prédictions dedc_power (la puissance en courant continu) et des autres variables.")
    
    st.markdown("1. Créer un modèle de machine learning pour prédire ac_power à partir des autres variables et des prédictions de dc_power.")
    df_nettoyage = df_nettoyage.copy()
    toutes_features = create_final_dataset(df_nettoyage)
    df_nettoyage["dc_power_pred"] = model_forest.predict(toutes_features)
    X_ac, y_ac = separer_X_ac_y_ac(df_nettoyage)
    X_train_ac, X_test_ac, y_train_ac, y_test_ac = split_train_test(X_ac, y_ac)
    model_ac = RandomForestRegressor(random_state=42)
    model_ac.fit(X_train_ac, y_train_ac)
    y_pred_ac = model_ac.predict(X_test_ac)
    
    st.markdown("2. Évaluer le modèle sur les données de test")
    df_score_ac = evaluer_modele(y_test_ac, y_pred_ac)
    
    st.markdown("3. Afficher les métriques de performance (ex: RMSE, MAE, R2)")
    st.dataframe(df_score_ac)
    
    st.markdown("4. Visualiser les prévisions du modèle sur les données de test")
    tracer_graphique_prediction_ac(y_test_ac, y_pred_ac)
    st.pyplot(plt)
    st.markdown("Le nuage de points confronte la puissance AC réellement mesurée à celle estimée par le modèle. la diagonale rouge représente l’accord parfait (x = y). La majorité des observations s’aligne près de cette diagonale, ce qui traduit une bonne corrélation générale entre prédiction et réalité")
    
    st.markdown("5. Comparer les prévisions avec les valeurs réelles")
    comparaison_ac = pd.DataFrame({"ac_power_reel": y_test_ac.values,"ac_power_pred": y_pred_ac})
    st.dataframe(comparaison_ac)
    st.markdown("Au regard des résultats obtenus, la production alternative prédite reste très proche de la production alternative réelle")
    
    st.markdown("7. Ajouter la prévision de ac_power dans le DataFrame df_to_pred_selected et visualisez l'évolution de la puissance en courant alternatif dans le temps.")
    df_to_pred_selected = df_to_pred_selected.copy()
    df_to_pred_selected["dc_power_pred"] = model_forest.predict(df_to_pred_selected)
    df_to_pred_selected_ac = df_to_pred_selected[["irradiation", "ambient_temperature", "Heure", "Jour", "dc_power_pred"]]
    df_to_pred_selected["ac_power_pred"] = model_ac.predict(df_to_pred_selected_ac)
    df_to_pred_selected["date_time"] = df_to_pred_features["date_time"].values
    st.dataframe(df_to_pred_selected.head())
    st.markdown("Visualisation de l'évolution de la puissance en courant alternatif dans le temps : ")
    tracer_evolution_prod_ac(df_to_pred_selected)
    st.pyplot(plt)
    st.markdown("Le tracé illustre la puissance AC que le modèle prévoit au fil du temps, sur la fin de journée la courbe reste nulle la nuit, puis s’élève dès l’aube, atteint un plateau vers midi avant de décliner en fin d’après‑midi, reproduisant le cycle solaire typique d’une centrale photovoltaïque. ")
    
    st.markdown("8. En déduire et visualiser l'efficacité de l'onduleur sur les données prédites (ratio entre la puissance en courant alternatif et la puissance en courant continu)")
    
    st.markdown("1. Créer une nouvelle variable efficacite_onduleur : ac_power / dc_power")
    
    df_to_pred_selected["efficacite_onduleur"] = (df_to_pred_selected["ac_power_pred"] / df_to_pred_selected["dc_power_pred"])
    st.dataframe(df_to_pred_selected.head())
    
    st.markdown("2. Visualiser l'évolution de cette variable dans le temps ainsi que celles de dc_power et ac_power")
    df_to_pred_selected["efficacité_pourcentage"] = df_to_pred_selected["efficacite_onduleur"] * 100
    tracer_evolution_efficacite_ondulateur(df_to_pred_selected)
    st.pyplot(plt)
    st.markdown("Le graphique superpose la puissance côté DC (bleu) issue des panneaux, la puissance côté AC (orange) délivrée par l’onduleur et, en pointillé vert, l’efficacité ondulateur. Nous avons volontairement exprimé l’efficacité de l’onduleur en pourcentage plutôt qu’en valeur décimale : tracée à côté des courbes de puissance, l’efficacité sous forme décimale paraissait visuellement nulle. La convertir en pourcentage ramène son amplitude dans une plage comparable à celle des puissances. On peut voir que l'efficacité ondulateur performe bien au moment de plein jour. En revanche, l'efficacité ondulatoire est nulle durant la nuit, c'est pourquoi il n'y a aucune courbe verte durant la nuit.")
    
    st.markdown("3. Analyser les variations de cette variable en fonction des conditions météorologiques")
    corr_eff = df_to_pred_selected[['efficacite_onduleur', 'ambient_temperature', 'irradiation']].corr()
    st.dataframe(corr_eff)
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_eff, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    st.pyplot(plt)
    st.markdown("D’après la matrice de corrélation, l’efficacité de l’onduleur ne dépend pratiquement pas des conditions météorologiques représentées ici : son coefficient de corrélation vaut seulement 0,03 avec la température ambiante et 0,10 avec la température des panneaux photovoltaïques, des valeurs proches de zéro qui indiquent une variation négligeable. ")
    
    
# --------------------------------------------------------------------------------------------------------------------
#                                      PARTIE RESIDUS
# -------------------------------------------------------------------------------------------------------------------- 

with tab7 : 
    st.subheader("Les résidus")
    st.markdown("1. Reprenez le modèle de régression linéaire de la question 4.2 et calculez les résidus (erreurs de prédiction) sur les données de test")                                                
    residus = y_test - y_pred
    df_residus = pd.DataFrame({"y_test": y_test.values, "y_pred": y_pred, "residuals": residus})
    st.dataframe(df_residus.head())
    
    st.markdown("2. Visualiser les résidus en fonction des valeurs prédites")
    tracer_visualisation_residus(y_pred,residus)
    st.pyplot(plt)
    st.markdown("Ce graphique montre les résidus entre la production réelle et la production que le modèle prédit. Idéalement, tous les points devraient se regrouper autour de la ligne rouge, résidu = 0, or on voit un schéma clair : pour les très faibles puissances prédites, l’erreur est plutôt négatif, pour les puissances moyennes à fortes, l’erreur devient positive. ")
        
        
    
    
    
     
     

   
     
     
     
     
     
     
     
     
   
    

