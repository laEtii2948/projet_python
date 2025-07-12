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
    
def lire_target_groupe_3(filename : str) -> pd.DataFrame | None : 
    
    try : 
        target_groupe_3 = pd.read_csv(filename)
        st.success(f"Fichier trouvé : {filename}")
        return target_groupe_3
    except Exception as e : 
        st.error(f"Fichier introuvable : {filename}")
        return None
    

def lire_plant_weather_forecast_groupe_3(filename : str) -> pd.DataFrame | None : 
    try : 
        plant_weather_forecast_groupe_3 = pd.read_csv(filename)
        st.success(f"Fichier trouvé : {filename}")
        return plant_weather_forecast_groupe_3
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
 
def create_final_dataset(df : pd.DataFrame) -> pd.DataFrame : 
    colonnes_selectionees = ["irradiation", "ambient_temperature", "Heure", "Jour"]
    return df[colonnes_selectionees].copy()

def separer_X_y(df : pd.DataFrame) -> pd.DataFrame | pd.Series : 
    X = create_final_dataset(df)
    y = df["dc_power"]
    st.info("Les données ont été séparées entre variables explicatives (X) et variable cible (y)")
    return X,y 


def split_train_test(X : pd.DataFrame, y : pd.Series, test_size : float = 0.2, random_state : int = 42) : 
    
    st.info("Les données ont été découpée en 80% de données d'entrainement et 20% de données de tests")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluer_modele_lineaire(y_test, y_pred) -> pd.DataFrame : 
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return pd.DataFrame({"MAE": [mae], "MSE": [mse], "RMSE": [rmse], "R²": [r2]})


def tracer_graphique_prediction(y_test, y_pred_prev, title : str = "Production réelle VS production prédite", xlabel : str = "Puissance DC réelle", ylabel : str = "Puissance DC prédite") -> None : 
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred_prev)
    # bissectrice
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="red")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def separer_X_ac_y_ac(df : pd.DataFrame) -> pd.DataFrame | pd.Series : 
    features_ac = ["irradiation","ambient_temperature","Heure","Jour","dc_power_pred"] 
    X_ac = df_nettoyage[features_ac].copy()
    y_ac = df_nettoyage["ac_power"].copy()
    st.info("Les données ont été séparées entre variables explicatives (X) et variable cible (y)")
    return X_ac, y_ac


def tracer_graphique_prediction_ac(y_test, y_pred_ac, title : str = "Production alternatif réelle VS production alternative prédite", xlabel : str = "Puissance AC réelle", ylabel : str = "Puissance AC prédite") -> None : 
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred_ac)
    # bissectrice
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="red")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def tracer_evolution_prod_ac(df : pd.DataFrame, title : str = "Évolution temporelle de la puissance AC (prédite)", xlabel : str = "Date", ylabel : str = "Puissance AC prédite") -> None : 
    plt.figure(figsize=(12, 6))
    plt.plot(df["date_time"], df["ac_power_pred"], label="AC prédite")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


    
    


    




    
    
        


st.set_page_config(
page_title="Projet python - Master MIMO - 2025",
page_icon=":python:",
layout="wide",
initial_sidebar_state="expanded",
)

st.header("Projet Python - Prédiction de la production solaire")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Présentation du projet","Exploration des données", "Préparation des données", "Importance des features et étude de corrélations entre les variables", "Modélisation", "Visualisation des résultats", "Bonus"])
with tab1:
    st.subheader("Présentation du projet")
    st.write("hello, coucoucdjfjdopjvopjfovfdvf")
with tab2:
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
    df_nettoyage = supprimer_module_temp(df_nouvelles_features)
    st.dataframe(df_nettoyage)
    st.markdown("4. Créer une fonction create_final_dataset(df) qui prend en entrée un DataFrame et renvoie un DataFrame avec les variables sélectionnées pour l'entraînement du modèle.")
    df_selected = create_final_dataset(df_nettoyage)
    st.dataframe(df_selected.head())
       
with tab5 : 
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
    df_score = evaluer_modele_lineaire(y_test, y_pred)
    st.success("Modèle évalué avec succès !")
    st.markdown("4. Afficher les métriques de performance (ex: RMSE, MAE, R2)")
    st.dataframe(df_score)
    st.markdown("5. Entraîner un modèle de régression par forêt aléatoire (RandomForestRegressor) sur les données d'entraînement")
    model_forest = RandomForestRegressor(random_state=42)
    model_forest.fit(X_train, y_train)
    st.success("Les données ont bien été entrainées avec le modèle de forêt aléatoire !")
    st.markdown("6. Évaluer le modèle sur les données de test")
    y_pred_random_forest = model_forest.predict(X_test)
    df_score_forest = evaluer_modele_lineaire(y_test, y_pred_random_forest)
    st.success("Modèle évalué avec succès !")
    st.markdown("7. Afficher les métriques de performance (ex: RMSE, MAE, R2)")
    st.dataframe(df_score_forest)
    st.markdown("8. Comparer les performances des deux modèles")
    df_comparer_modeles = pd.concat([df_score, df_score_forest])
    st.dataframe(df_comparer_modeles)
    st.markdown("Au vu des résultats obtenus, le modèle de forêt aléatoire semble donner de meilleurs résultats")
      
with tab6 : 
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

with tab7 : 
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
    df_score_ac = evaluer_modele_lineaire(y_test_ac, y_pred_ac)
    st.markdown("3. Afficher les métriques de performance (ex: RMSE, MAE, R2)")
    st.dataframe(df_score_ac)
    st.markdown("4. Visualiser les prévisions du modèle sur les données de test")
    tracer_graphique_prediction_ac(y_test_ac, y_pred_ac)
    st.pyplot(plt)
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
    st.markdown("8. En déduire et visualiser l'efficacité de l'onduleur sur les données prédites (ratio entre la puissance en courant alternatif et la puissance en courant continu)")
    st.markdown("1. Créer une nouvelle variable efficacite_onduleur : ac_power / dc_power")
    df_to_pred_selected["efficacite_onduleur"] = (df_to_pred_selected["ac_power_pred"] / df_to_pred_selected["dc_power_pred"])
    st.dataframe(df_to_pred_selected.head())
    st.markdown("2. Visualiser l'évolution de cette variable dans le temps ainsi que celles de dc_power et ac_power")
    
    df_to_pred_selected["efficacité_pourcentage"] = df_to_pred_selected["efficacite_onduleur"] * 100

    plt.figure(figsize=(14, 6))
    plt.plot(df_to_pred_selected["date_time"],
         df_to_pred_selected["dc_power_pred"], label="🔋 DC Power")
    plt.plot(df_to_pred_selected["date_time"],
         df_to_pred_selected["ac_power_pred"], label="⚡ AC Power")
    plt.plot(df_to_pred_selected["date_time"],
         df_to_pred_selected["efficacité_pourcentage"] * 100, "--", label="⚙️ Efficacité [%]")
    plt.xlabel("Date-heure"); plt.ylabel("Valeur (W ou %)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    st.pyplot(plt)
        
        
    
    
    
     
     

   
     
     
     
     
     
     
     
     
   
    

