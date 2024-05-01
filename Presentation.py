import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import folium
import warnings
warnings.filterwarnings('ignore')
#from streamlit_folium import folium_static


# Configuration de la page
st.set_page_config(
    page_title="Vélos à Paris",
    page_icon="🚲",
)

# Titre principal
st.title("🚲 :blue[Trafic cycliste à Paris]")

# Charger le fichier original pour le mettre en cache
@st.cache_data
def load_data(fichier):
    df = pd.read_csv(fichier,sep=';') # Download the data
    return df
df= load_data('comptage velo.csv')

# Charger le fichier corrigée pour le mettre en cache
@st.cache_data
def load_data1(fichier):
    df = pd.read_csv(fichier) # Download the data
    return df

df_corrected = load_data1('comptage velo corrected.csv')
# Convertir la colonne 'Date comptage' en datetime si elle n'est pas déjà de ce type
df_corrected['Date comptage']= pd.to_datetime(df_corrected['Date comptage'])
df_corrected["Date installation"]= pd.to_datetime(df_corrected["Date installation"])


#df = pd.read_csv("comptage velo corrected.csv")

# Convertir la colonne 'Date comptage' en datetime si elle n'est pas déjà de ce type
#df['Date comptage'] = pd.to_datetime(df['Date comptage'])
    
# Extraire la date sans l'heure
df_corrected['Date'] = df_corrected['Date comptage'].dt.date
    
# Agréger les données par jour
daily_sum = df_corrected.groupby('Date').agg({'Comptage horaire': 'sum'}).reset_index()


# Barre latérale avec les options de page
st.sidebar.title("Sommaire")
pages = ["Introduction", "Exploration des données 🌍", "Visualisation 📊", "Modélisation 💻", "Conclusion"]
page = st.sidebar.radio("Etapes", pages)

#Nom des contributeurs
st.sidebar.title("Contributeurs")
st.sidebar.markdown("[Cintyha Dina](https://www.linkedin.com/in/cintyha-dina-98396a97/)")
st.sidebar.markdown("[Pascal Paineau](https://www.linkedin.com/in/papaineau72/)")
st.sidebar.markdown("[Stephane Moisan](https://www.linkedin.com/in/stephanemoisan)")

# Logique pour afficher différentes pages
if page == "Introduction" :
    st.write("### :blue[Introduction]👋")
    st.image("paris-velo.jpg")
    
    Texte="""Le projet que nous vous présentons aujourd'hui est une analyse des données collectées à partir des compteurs à vélo permanents que la Ville de Paris déploie depuis plusieurs années déjà afin d’évaluer le développement de la pratique cycliste."""
    st.markdown(f"<div style='text-align: justify'>{Texte}</div>", unsafe_allow_html=True)
    st.markdown(""" """)
    st.markdown(
        """
        
        **👈 Vous trouverez sur le volet à gauche les différentes étapes de notre analyse.**
        
        ### :blue[Objectifs]
        
        """
    )
    Texte="""Notre principal objectif est de donner une vue d'ensemble de l'usage du vélo à Paris et de faire émerger des tendances afin d'aider à la prise de décision quant à l'aménagement des pistes cyclables et les possibilités d'investissement dans ce moyen de transport moins polluant que la voiture."""
    st.markdown(f"<div style='text-align: justify'>{Texte}</div>", unsafe_allow_html=True)


#-------------------------------------------------------------------------------------------------------------------
#EXPLORATION DES DONNEES
elif page == "Exploration des données 🌍":
    st.write("### :blue[Exploration des données]🌍")

    # 1ère partie ---------
    exploration_donnees = st.radio("**Différentes étapes d'analyse**", ("Fichier source", "Exploration","Préparation", "Fichier de travail"), index=None)
    if exploration_donnees == "Fichier source":
        st.write("##### :blue[Présentation du jeu de données et mode de collecte]")
        
        Texte="""Dans le cadre du projet Data Analyst, nous avons utilisé un jeu de données sur 12 mois glissants 
        (après retraitement) du 01/07/2022 au 30/06/2023 à partir du fichier comptage-velo-donnees-compteurs.csv 
        extrait en date du 02/08/2023. """
        st.markdown(f"<div style='text-align: justify'>{Texte}</div>", unsafe_allow_html=True)
        st.markdown(""" """)
        st.markdown(
        """
        
    
        Ces données sont disponibles sur le site «Paris data» en open source sur le lien suivant :
         https://opendata.paris.fr/explore/dataset/comptage-velo-donneescompteurs/information/?disjunctive.id_compteur&disjunctive.nom_compteur&disjunctive.id&disjunctive.name 
     
    """
    )
        Texte="""Le jeu de données est chargé quotidiennement sur l’API du partenaire Eco Compteur à Paris et évolue chaque jour à J-1.

Le fichier csv comporte un jeu de données des comptages horaires de vélos par compteur et localisation des sites de comptage. 

Un site de comptage peut être équipé d’un compteur dans le cas d’un aménagement cyclable unidirectionnel ou de deux compteurs dans le cas d’un aménagement cyclable bidirectionnel. La Ville de Paris déploie depuis plusieurs années des compteurs à vélo permanents pour évaluer le développement de la pratique cycliste. 

Les compteurs sont situés sur des pistes cyclables et dans certains couloirs de bus ouverts aux vélos.

"""
        st.markdown(f"<div style='text-align: justify'>{Texte}</div>", unsafe_allow_html=True)


    if exploration_donnees == "Exploration":
        
        st.write("Affichage du jeu de données")
        st.dataframe(df.head())
    
       
        if st.checkbox("Afficher le data shape"):
                st.write(df.shape)
            
        elif st.checkbox("Afficher le descriptif"):
                st.write(df.describe())
    
        elif st.checkbox("Type"):
                st.write(df.dtypes)
    
        elif st.checkbox("Valeurs manquantes"):
                Val = df.isna.sum()/(len(df)*100)
                st.dataframe(Val)
    
        elif st.checkbox("Doublons"):
                Val=df.duplicated().sum()
                st.write("Nombre de doublons :", Val )
    
        elif st.checkbox("Caractères spéciaux"):
                st.write("Les caractéres spéciaux à chercher sont : .*[@_!#$%^&*()<>?/\|}{~:].*")
                st.write((df["Nom du compteur"][df["Nom du compteur"].str.contains(".*[@_!#$%^&*()<>?/\|}{~:].*",regex=True)].unique()))
    
    if exploration_donnees == "Préparation":
        
        st.write("Affichage du jeu de données df_corrected")
        st.dataframe(df_corrected.head())
        
        if st.checkbox("Type df_corrected"):
                st.write(df_corrected.dtypes)
        
    if exploration_donnees == "Fichier de travail":
        
        st.write("Affichage du jeu de données df_corrected")
        st.dataframe(df_corrected.head())
        
        if st.checkbox("Afficher le data shape du jeu de données df_corrected"):
                st.write(df_corrected.shape)
                
            
        elif st.checkbox("Afficher le descriptif du jeu modifié"):
                st.dataframe(df_corrected.describe())


#-------------------------------------------------------------------------------------------------------------------
#VISUALISATION
elif page == "Visualisation 📊":
    st.write("### :blue[Visualisation des données]📊")
    st.markdown("### Présentation du trafic cycliste à Paris")
    
    selectbox = st.selectbox ("**Eléments d'analyse visuelle :**", ("Choisir un graphique", "Evolution de l'installation des compteurs par année", "Evolution du nombre de kilomètres aménagés","Répartition des compteurs dans la ville", "Affichage des compteurs selon le nombre de passage"))
    if selectbox == "Evolution de l'installation des compteurs par année" :
        # Convertir les colonnes de dates en type datetime
        df_corrected['Date comptage'] = pd.to_datetime(df_corrected['Date comptage'])
        df_corrected['Date installation'] = pd.to_datetime(df_corrected['Date installation'])

        # Extraire l'année d'installation des compteurs
        df_corrected['Annee installation'] = df_corrected['Date installation'].dt.year

        # Compter le nombre de compteurs installés par année
        counts_by_year = df_corrected['Annee installation'].value_counts().sort_index()

        # Visualiser le nombre de compteurs installés par année
        st.write("Nombre de compteurs installés chaque année :")
        st.bar_chart(counts_by_year)


    if selectbox == "Evolution du nombre de kilomètres aménagés":
        Ev_res = {
            "Année": [2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
            "Linéaire en km": [292.8, 327.3, 370.9, 399.3, 439.5, 446.2, 647.5, 654.8, 677, 732.5, 737.5, 742.1, 779.8, 835.6, 912.6, 1037.05, 1136.3, 1170.8, 1202.5]
        }
        df_piste = pd.DataFrame(Ev_res)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        df_piste.plot(kind="bar", x="Année", y="Linéaire en km", width=0.7, color="blue", ax=ax)
        
        ax.set_title("Evolution du nombre de kilomètres aménagés", fontsize=14, fontweight='bold')
        ax.set_xlabel('Année')
        ax.set_ylabel('Nombre de kilomètres aménagés')
        ax.tick_params(axis='x', rotation=45)
        
        st.pyplot(fig) 

    
   

    if selectbox == "Répartition des compteurs dans la ville":
        df_plan = df_corrected.drop_duplicates(subset="Nom du compteur", keep='first')

        # Créer la carte avec Folium
        paris = folium.Map(location=[48.856578, 2.351828],zoom_start=12, min_zoom=12, max_zoom=18)

        # Ajouter une couche
        folium.TileLayer('openstreetmap').add_to(paris)

        # Ajouter un marqueur pour chaque compteur
        for index, row in df_plan.iterrows():
            folium.Marker(location=[row["Latitude"], row["Longitude"]],
                        popup=row["Nom du compteur"],
                        tooltip="Cliquez ici pour voir le nom du compteur").add_to(paris)

        # Ajouter un contrôle de couches à la carte
        folium.LayerControl().add_to(paris)

        # Afficher la carte Folium dans Streamlit
        folium_static(paris)


    

    if selectbox == "Affichage des compteurs selon le nombre de passage":
        # Créer la carte avec Folium
        paris = folium.Map(location=[48.856578, 2.351828], zoom_start=12, min_zoom=12, max_zoom=18)

        # Grouper les données par compteur et calculer la somme des passages
        df_plan_sum = df_corrected.groupby(['Nom du compteur', "Latitude", "Longitude"])['Comptage horaire'].sum().reset_index()

        # Définir la couleur en fonction du nombre de passages
        def colorer_zone(comptage):
            if comptage > seuil:
                return 'red'  # zones avec plus de passages en rouge
            else:
                return 'green'  # zones avec moins de passages en vert

        # Définir le seuil pour différencier les zones avec plus ou moins de passages
        seuil = df_plan_sum['Comptage horaire'].quantile(0.75)  # par exemple, seuil à 75ème percentile

        # Ajouter des marqueurs de cercle pour chaque zone avec couleur basée sur le nombre de passages
        for index, row in df_plan_sum.iterrows():
            folium.CircleMarker(location=[row["Latitude"], row["Longitude"]],
                                popup=row["Comptage horaire"],
                                radius=row["Comptage horaire"] / 100000,
                                fill=True,
                                color=colorer_zone(row["Comptage horaire"]),
                                tooltip="Cliquez ici pour voir le nombre de passage").add_to(paris)

        # Afficher la carte Folium dans Streamlit
        folium_static(paris)



#Visualisation du trafic en fonction de la temporalité
    temporalite = st.radio('**Nombre de passage des vélos sur quelle période ?**', ('Jour', 'Semaine', 'Mois', 'Année'), index=None)
    if temporalite == 'Jour':
        df_graph = df_corrected.copy()
        df_graph['Heure comptage']=(df_graph['Date comptage'].dt.time)
        df_graph = df_graph[['Heure comptage', 'Comptage horaire']]
        df_compt_hour = df_graph.groupby("Heure comptage").mean()
        fig, ax = plt.subplots(figsize=(10, 6))
        df_compt_hour.plot(kind='bar', color='green', ax=ax)
        plt.title("Sur une journée", fontsize=14, fontweight='bold')
        plt.xlabel('Heure de comptage')
        plt.ylabel('Moyenne des passages')
        plt.xticks(rotation=45)
        plt.tight_layout()  
        st.pyplot(fig)

    if temporalite == "Mois" :
        df_graph = df_corrected.copy()
        df_graph = df_graph[(df_graph['Date comptage']>='2023-04-01') & (df_graph['Date comptage']<'2023-05-01')]
        hebdo_sum = df_graph.groupby(df_graph['Date comptage'].dt.to_period("d")).agg({'Comptage horaire': 'sum'}).reset_index()
        fig = plt.figure(figsize=(10, 6))
        x=hebdo_sum['Date comptage'].dt.start_time
        y=hebdo_sum['Comptage horaire']
        plt.plot(x, y,linestyle='-', marker='o')
        plt.title("Sur le mois d'avril 2023",fontsize=14,fontweight='bold')
        plt.xlabel('Jour')
        plt.ylabel('Nombre de comptage')
        plt.xticks(rotation=45)
        plt.annotate(
            'Week-end',fontweight='bold',color='green', xy=(110, 140), xytext=(120, 100),xycoords='figure points',
                    arrowprops=dict(facecolor='green', shrink=0.05))
        plt.annotate(
            '',fontweight='bold',color='green', xy=(250, 100), xytext=(180, 105),xycoords='figure points',
                    arrowprops=dict(facecolor='green', shrink=0.05))
        plt.annotate(
            'Grèves',fontweight='bold',color='red', xy=(290, 260), xytext=(280, 110),xycoords='figure points',
                    arrowprops=dict(facecolor='red', shrink=0.05))
        plt.annotate(
            '',fontweight='bold',color='red', xy=(335, 180), xytext=(320, 125),xycoords='figure points',
                    arrowprops=dict(facecolor='red', shrink=0.05))
        plt.annotate(
            '',fontweight='bold',color='red', xy=(480, 90), xytext=(320, 110),xycoords='figure points',
                    arrowprops=dict(facecolor='red', shrink=0.05))
        plt.annotate(
            'Week-end',fontweight='bold',color='green', xy=(340, 190), xytext=(390, 170),xycoords='figure points',
                    arrowprops=dict(facecolor='green', shrink=0.05))
        plt.annotate(
            'Week-end',fontweight='bold',color='green', xy=(480, 100), xytext=(520, 100),xycoords='figure points',
                    arrowprops=dict(facecolor='green', shrink=0.05))
        st.pyplot(fig)


    if temporalite == "Année":
        
        df_graph = df_corrected.copy()
        df_graph['Jour'] = df_graph['Date comptage'].dt.strftime('%Y-%m-%d')
        df_graph['Mois'] = df_graph['Date comptage'].dt.strftime('%Y-%m')
        monthly_sum = df_graph.groupby('Mois')['Comptage horaire'].sum().reset_index()

        # Créer un graphique à barres
        fig, ax = plt.subplots(figsize=(10, 6))
        x = monthly_sum['Mois']
        y = monthly_sum['Comptage horaire']
        ax.bar(x, y, width=0.9, alpha=0.5, color=['blue', 'blue', 'blue', 'green', 'green', 'green', 'yellow', 'yellow', 'yellow', 'orange', 'orange', 'orange'])
        
        def addlabels(x,y):
            for i in range(len(x)):
                plt.text(i, y[i], y[i], ha='center',
                bbox=dict(facecolor='red', alpha=.2), fontsize=8, fontweight='bold')

        addlabels(x, y)
        plt.title('Sur une année', fontweight='bold')
        plt.xlabel('Mois')
        plt.ylabel('Nombre de comptage')
        plt.xticks(rotation=45)

        # Ajouter des annotations avec des flèches
        ax.annotate('Vacances de Noël', fontweight='bold', color='royalblue', xy=(290, 220), xytext=(300, 300), xycoords='figure points',
                    arrowprops=dict(facecolor='fuchsia', shrink=0.05))
        ax.annotate('Vacances d\'été', fontweight='bold', color='royalblue', xy=(120, 250), xytext=(70, 350), xycoords='figure points',
                    arrowprops=dict(facecolor='fuchsia', shrink=0.05))
        ax.annotate('Jours fériés et ponts', fontweight='bold', color='royalblue', xy=(460, 310), xytext=(380, 350), xycoords='figure points',
                    arrowprops=dict(facecolor='fuchsia', shrink=0.05))

        st.pyplot(fig)

    if temporalite == "Semaine":
        df_graph = df_corrected.copy()
        
        # Filtrer les données pour ne considérer que la première semaine de juin
        first_week_june = df_graph[(df_graph['Date comptage'].dt.month == 6) & (df_graph['Date comptage'].dt.day >= 1) & (df_graph['Date comptage'].dt.day <= 7)]
        
        # Grouper par jour et calculer le nombre total de passages de vélos pour chaque jour
        passages_per_day = first_week_june.groupby(df_graph['Date comptage'].dt.dayofweek).agg({'Comptage horaire': 'sum'}).reset_index()
        
        # Convertir le numéro du jour en nom complet du jour de la semaine en français
        jours_semaine_fr = {0: 'Lundi', 1: 'Mardi', 2: 'Mercredi', 3: 'Jeudi', 4: 'Vendredi', 5: 'Samedi', 6: 'Dimanche'}
        passages_per_day['Jour de la semaine'] = passages_per_day['Date comptage'].apply(lambda x: jours_semaine_fr[x])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(passages_per_day['Jour de la semaine'], passages_per_day['Comptage horaire'], color='blue')
        ax.set_title('Sur la première semaine de juin', fontsize=14, fontweight='bold')
        ax.set_xlabel('Jours de la semaine')
        ax.set_ylabel('Nombre de passages de vélos')
        ax.tick_params(axis='x', rotation=45)  # Rotation des dates pour une meilleure lisibilité
        
        st.pyplot(fig)


#-------------------------------------------------------------------------------------------------------------------
#MODELISATION
elif page == "Modélisation 💻":
    st.write("### :blue[Modélisation]💻")
    
    st.markdown("### :black[Choix du modèle]")
    st.markdown("""Notre choix s'est porté sur un modéle de :blue[Séries temporelles] du fait que nous avons observé une forte dépendance temporelle dans ce dataframe.""")
    st.divider()
    
    presentation_modelisation = st.radio('**Nombre de passage des vélos sur quelle période ?**', ("Décomposition", "Choix paramètres","Modélisation"), index=None)

    if presentation_modelisation == "Décomposition" :
        st.markdown("##### :black[Décomposition]")
        df_corrected['Date comptage']= pd.to_datetime(df_corrected['Date comptage'])
        velo_decomp = df_corrected.copy()
        velo_decomp['Jour'] = velo_decomp['Date comptage'].dt.strftime('%Y-%m-%d')
        velo_decomp = velo_decomp.groupby('Jour')['Comptage horaire'].sum().reset_index()
        velo_decomp = velo_decomp.sort_values(by = 'Jour')
        velo_decomp['Jour']= pd.to_datetime(velo_decomp['Jour'])
        velo_decomp.set_index('Jour', inplace=True)
        
        option = st.selectbox(
       "Choix de la décomposition ?",
       ("Décomposition Additif", "Décomposition Multiplicatif"),
       index=None,
       placeholder="Selection de la décomposition...",
    )

        
        if option =="Décomposition Additif":
            
            st.markdown("##### :blue[Décomposition additif]")
            st.divider()
            
            #Décomposition additif
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(velo_decomp)
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            fig = plt.figure(figsize=(20, 6))
            plt.plot(trend)
            plt.title('Tendance',fontsize=26,fontweight='bold')
            st.pyplot(fig)
            
            fig = plt.figure(figsize=(20, 6))
            plt.plot(seasonal)
            plt.title('Saisonnalité',fontsize=26,fontweight='bold')
            st.pyplot(fig)
            
            fig = plt.figure(figsize=(20, 6))
            plt.plot(residual,"o")
            plt.title('Résidus',fontsize=26,fontweight='bold')
            st.pyplot(fig)
            
            fig = plt.figure(figsize=(20, 6))
            pd.plotting.autocorrelation_plot(velo_decomp)
            plt.title('Autocorrelation',fontsize=26,fontweight='bold')
            st.pyplot(plt.gcf())
            

            from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
            fig,(ax1,ax2) = plt.subplots(2,1,figsize=(20,6))
            plot_acf(velo_decomp['Comptage horaire'], lags=30, zero=True, ax=ax1)
            ax1.set_title('ACF - Série de comptage horaire',fontsize=26,fontweight='bold')
            ax1.set_xlabel('Lag')
            ax1.set_ylabel('Corrélation')
            ax1.grid(True)
            
            plot_pacf(velo_decomp['Comptage horaire'], lags=30, zero=True, ax=ax2)
            ax2.set_title('PACF - Série de comptage horaire',fontsize=26,fontweight='bold')
            ax2.set_xlabel('Lag')
            ax2.set_ylabel('Corrélation partielle')
            ax2.grid(True)
            plt.tight_layout()
            st.pyplot(fig)
            
            from statsmodels.tsa.stattools import adfuller
            result=adfuller(velo_decomp)
            st.write ("Valeur de test : ",(result[0]))  
            st.write ("P-valeur : " ,(result[1])) 
            st.markdown ("Conclusion : :red[La série n'est pas stationnaire]")
            
        elif option =="Décomposition Multiplicatif":
            
            st.markdown("##### :blue[Décomposition Multiplicatif]")
            st.divider()
            
            #Décomposition multiplicative
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomposition = seasonal_decompose(velo_decomp, model = 'multiplicative')
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            fig = plt.figure(figsize=(20, 6))
            plt.plot(trend)
            plt.title('Tendance',fontsize=26,fontweight='bold')
            st.pyplot(fig)
            
            fig = plt.figure(figsize=(20, 6))
            plt.plot(seasonal)
            plt.title('Saisonnalité',fontsize=26,fontweight='bold')
            st.pyplot(fig)
            
            fig = plt.figure(figsize=(20, 6))
            plt.plot(residual,"o")
            plt.title('Résidus',fontsize=26,fontweight='bold')
            st.pyplot(fig)


            st.markdown("##### :blue[Retour à un modéle additif]")
            st.divider()
            velo_decomp_log = np.log(velo_decomp)
            st.line_chart(velo_decomp_log)
            plt.xticks(rotation=45)
            
            st.markdown("##### :blue[Choix différenciation]")
            st.divider()
            col1, col2 = st.columns(2)
            col1.write("**Différenciation simple**")
            col2.write("**Différenciation d'ordre 7**")
            
            fig = plt.subplots(1, 2, figsize=(10,6))
            plt.title("Différenciation simple",fontsize=18,fontweight='bold')
            velo_decomp_log_1 = velo_decomp_log.diff().dropna()
            col1.line_chart(velo_decomp_log_1)
            
            fig = plt.subplots(1, 2, figsize=(10,6))
            plt.title("Différenciation d'ordre 7",fontsize=18,fontweight='bold')
            velo_decomp_log_2 = velo_decomp_log_1.diff(periods = 7).dropna()
            col2.line_chart(velo_decomp_log_2)
            
            fig = plt.subplots(figsize=(10,6))
            pd.plotting.autocorrelation_plot(velo_decomp_log_1)
            plt.title("Autocorrelation simple",fontsize=18,fontweight='bold')
            col1.pyplot(plt.gcf())
            
            fig = plt.subplots(figsize=(10,6))
            pd.plotting.autocorrelation_plot(velo_decomp_log_2)
            plt.title("Autocorrelation d'ordre 7",fontsize=18,fontweight='bold')
            col2.pyplot(plt.gcf())
            
            from statsmodels.tsa.stattools import adfuller
            result=adfuller(velo_decomp_log_2)
            st.write ('Valeur de test : ', result[0])
            st.write ('P-valeur : ', result[1])
            st.markdown ("Conclusion : :green[La série est stationnaire]")
                                   
    if presentation_modelisation == "Choix paramètres":
                        
        df_corrected['Date comptage']= pd.to_datetime(df_corrected['Date comptage'])
        velo_decomp = df_corrected.copy()
        velo_decomp['Jour'] = velo_decomp['Date comptage'].dt.strftime('%Y-%m-%d')
        velo_decomp = velo_decomp.groupby('Jour')['Comptage horaire'].sum().reset_index()
        velo_decomp = velo_decomp.sort_values(by = 'Jour')
        velo_decomp['Jour']= pd.to_datetime(velo_decomp['Jour'])
        velo_decomp.set_index('Jour', inplace=True)
        velo_decomp_log = np.log(velo_decomp)
        velo_decomp_log_1 = velo_decomp_log.diff().dropna()
        velo_decomp_log_2 = velo_decomp_log_1.diff(periods = 7).dropna()
        
        st.markdown("##### :blue[Etude ACF et PACF avec différenciation d'ordre 7]")
        from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(20,6))
        plot_acf(velo_decomp_log_2['Comptage horaire'], lags=30, zero=True, ax=ax1)
        ax1.set_title('ACF - Série de comptage horaire',fontsize=26,fontweight='bold')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Corrélation')
        ax1.grid(True)
        
        plot_pacf(velo_decomp_log_2['Comptage horaire'], lags=30, zero=True, ax=ax2)
        ax2.set_title('PACF - Série de comptage horaire',fontsize=26,fontweight='bold')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Corrélation partielle')
        ax2.grid(True)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("##### :blue[Choix paramètres]")
        Texte = """  Pour la partie non saisonnière, on en a estimé que le un processus  ARMA(𝑝,𝑞) serait ARMA(1,1)."""
        st.markdown(""":green[ORDER] :""")
        st.markdown(f"<div style='text-align: justify'>{Texte}</div>", unsafe_allow_html=True)  

        Texte="""Pour les ordres saisonniers (𝑃 et 𝑄), il suffit de se référer aux mêmes règles mais en regardant uniquement les pics saisonniers (les 7ème, 14ème, 21ème, 28ème pics). Ainsi on peut estimer pour la partie saisonnière un processus MA(1) ce qui est équivalent à un  ARMA(0,1)"""
        st.markdown(""" """)
        st.markdown(""":green[SEASONAL ORDER] :""")
        st.markdown(f"<div style='text-align: justify'>{Texte}</div>", unsafe_allow_html=True)
        
        Texte = """* Une décroissance de l'ACF et la PACF sans coupure nette : modèle  ARMA(1,1), (Il est possible de rajouter des termes si le modèle ne semble pas suffisamment performant)
* Pour les pics saisonniers, une coupure de l'ACF après la première période et une décroissance de la PACF : modèle  MA(1).
* Et une décomposition d'ordre 7."""
        st.markdown(""" """)
        st.markdown(""":green[POUR CONCLURE] :""")
        #st.markdown(f"<div style='text-align: justify'>{Texte}</div>", unsafe_allow_html=True)
        st.markdown(Texte)
        st.markdown("""Nous entrainerons donc un modèle :blue[SARIMA(1,1,1)(0,1,1)7]""")

        velo_decomp = df_corrected.copy()
        velo_decomp['Jour'] = velo_decomp['Date comptage'].dt.strftime('%Y-%m-%d')
        velo_decomp = velo_decomp.groupby('Jour')['Comptage horaire'].sum().reset_index()
        velo_decomp = velo_decomp.sort_values(by = 'Jour')
        velo_decomp['Jour']= pd.to_datetime(velo_decomp['Jour'])
        velo_decomp.set_index('Jour', inplace=True)

        #Séparer mon jeu de données en jeu d'entrainement et jeu de test
        train_data = velo_decomp[:round(len(velo_decomp)*70/100)]
        test_data = velo_decomp[round(len(velo_decomp)*70/100):]

        Tableau = st.checkbox('Affichage SARIMAX')

        if Tableau:

            #Sarima2
            model=sm.tsa.SARIMAX(train_data,order=(1,1,1),seasonal_order=(0,1,1,7))
            sarima2=model.fit()
            st.write(sarima2.summary())
        
    if presentation_modelisation == "Modélisation":

        velo_decomp = df_corrected.copy()
        velo_decomp['Jour'] = velo_decomp['Date comptage'].dt.strftime('%Y-%m-%d')
        velo_decomp = velo_decomp.groupby('Jour')['Comptage horaire'].sum().reset_index()
        velo_decomp = velo_decomp.sort_values(by = 'Jour')
        velo_decomp['Jour']= pd.to_datetime(velo_decomp['Jour'])
        velo_decomp.set_index('Jour', inplace=True)

        #Séparer mon jeu de données en jeu d'entrainement et jeu de test
        train_data = velo_decomp[:round(len(velo_decomp)*70/100)]
        test_data = velo_decomp[round(len(velo_decomp)*70/100):]

        st.markdown("##### :blue[Modélisation]")
        st.markdown("Nous avons aussi procéder à une recherche automatique pour trouver les meilleurs paramètres à prendre en compte. Il nous a été proposé le modèle :blue[SARIMA(1,0,1)(0,1,1)7]")
        st.divider()
        
        #Sarima2
        model=sm.tsa.SARIMAX(train_data,order=(1,0,1),seasonal_order=(0,1,1,7))
        sarima2=model.fit()
        
        train_predictions = sarima2.predict(start = train_data.index[0], end = train_data.index[-1])
        test_predictions = sarima2.predict(start = test_data.index[0], end = test_data.index[-1])
        
        col1, col2 = st.columns(2)
        col1.write("**Modèle proposé avec order=(1,0,1)**")
        col2.write("**Modèle déduit avec order=(1,1,1)**")
    
        #Calculer les différents métriques
        from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
        #Mesure la déviation absolue moyenne entre une estimation prévue et les données réelles. - np.mean(abs(diff)))
        train_mae = mean_absolute_error(train_data,train_predictions)
        #La distance, entre la prévision et l’observation, est ici élevée au carré. La sensibilité à l’erreur est meilleure - np.mean(diff**2)
        train_mse = mean_squared_error(train_data,train_predictions)
        #C'est la racine carrée du MSE, c’est une métrique largement utilisée - np.sqrt(mse)
        train_rsme = mean_squared_error(train_data,train_predictions, squared=False)
        #1-(sum(diff**2)/sum((y_true-np.mean(y_true))**2))
        train_r2 = r2_score(train_data, train_predictions)
        
        test_mae = mean_absolute_error(test_data,test_predictions)
        test_mse = mean_squared_error(test_data,test_predictions)
        test_rsme = mean_squared_error(test_data,test_predictions, squared=False)
        test_r2 = r2_score(test_data, test_predictions)
        
        performance_df = pd.DataFrame({
            'Métrique' : ['MAE', 'MSE','RMSE', 'R2'],
            'Ensemble (Entrainement)' : (train_mae, train_mse, train_rsme, train_r2),
            'Ensemble (Test)' : (test_mae, test_mse, test_rsme, test_r2)})
        col1.dataframe(performance_df)
    
    #Visualisation du modéle sur les jeux d'entrainements et de tests
        from datetime import date

        
        fig=plt.figure(figsize=(10,6))
        plt.plot(train_data.index, train_data.values, label='Ensemble (Entrainement)', color='red')
        plt.plot(test_data.index, test_data.values, label='Ensemble (Test)', color='green')
        plt.plot(train_predictions.index, train_predictions.values, label='Prédictions (Entrainement)', color='blue', linestyle='--')
        plt.plot(test_predictions.index, test_predictions.values, label='Prédictions (Test)', color='blue', linestyle='--')
        
        plt.xlim(train_data.index[0], test_data.index[-1])

        plt.xlabel('Date')
        plt.ylabel('comptage')
        plt.title("Choix du modéle SARIMA",fontweight='bold')
        plt.xticks(rotation=45)
        plt.axvline(x= date(2023,3,15), color='black', linestyle='--')
        plt.legend()
        plt.annotate('Entrainement',fontweight='bold',color='black', xy=(480, 100), xytext=(380, 100),xycoords='figure points')
        plt.annotate('Test',fontweight='bold',color='black', xy=(480, 100), xytext=(520, 100),xycoords='figure points')
        plt.show();
        col1.pyplot(fig)
    
        #Sarima1
        model1=sm.tsa.SARIMAX(train_data,order=(1,1,1),seasonal_order=(0,1,1,7))
        sarima1=model1.fit()
        
        train_predictions = sarima1.predict(start = train_data.index[0], end = train_data.index[-1])
        test_predictions = sarima1.predict(start = test_data.index[0], end = test_data.index[-1])
    
        #Calculer les différents métriques
        #Mesure la déviation absolue moyenne entre une estimation prévue et les données réelles. - np.mean(abs(diff)))
        train_mae = mean_absolute_error(train_data,train_predictions)
        #La distance, entre la prévision et l’observation, est ici élevée au carré. La sensibilité à l’erreur est meilleure - np.mean(diff**2)
        train_mse = mean_squared_error(train_data,train_predictions)
        #C'est la racine carrée du MSE, c’est une métrique largement utilisée - np.sqrt(mse)
        train_rsme = mean_squared_error(train_data,train_predictions, squared=False)
        #1-(sum(diff**2)/sum((y_true-np.mean(y_true))**2))
        train_r2 = r2_score(train_data, train_predictions)
        
        test_mae = mean_absolute_error(test_data,test_predictions)
        test_mse = mean_squared_error(test_data,test_predictions)
        test_rsme = mean_squared_error(test_data,test_predictions, squared=False)
        test_r2 = r2_score(test_data, test_predictions)
        
        performance_df = pd.DataFrame({
            'Métrique' : ['MAE', 'MSE','RMSE', 'R2'],
            'Ensemble (Entrainement)' : (train_mae, train_mse, train_rsme, train_r2),
            'Ensemble (Test)' : (test_mae, test_mse, test_rsme, test_r2)})
        col2.dataframe(performance_df)
    
        #Visualisation du modéle sur les jeux d'entrainements et de tests
        from datetime import date
        fig=plt.figure(figsize=(10,6))
        plt.plot(train_data.index, train_data.values, label='Ensemble (Entrainement)', color='red')
        plt.plot(test_data.index, test_data.values, label='Ensemble (Test)', color='green')
        plt.plot(train_predictions.index, train_predictions.values, label='Prédictions (Entrainement)', color='blue', linestyle='--')
        plt.plot(test_predictions.index, test_predictions.values, label='Prédictions (Test)', color='blue', linestyle='--')
        
        plt.xlim(train_data.index[0], test_data.index[-1])
    
        plt.xlabel('Date')
        plt.ylabel('comptage')
        plt.title("Choix du modéle SARIMA",fontweight='bold')
        plt.xticks(rotation=45)
        plt.axvline(x= date(2023,3,15), color='black', linestyle='--')
        plt.legend()
        plt.annotate('Entrainement',fontweight='bold',color='black', xy=(480, 100), xytext=(380, 100),xycoords='figure points')
        plt.annotate('Test',fontweight='bold',color='black', xy=(480, 100), xytext=(520, 100),xycoords='figure points')
        plt.show();
        col2.pyplot(fig)
    
        Prediction = st.checkbox('Prédiction')

        futureDate = pd.DataFrame(pd.date_range(start='2023-07-01', end='2023-09-30',freq='D'),columns=['Dates'])
        futureDate.set_index('Dates',inplace=True)
        
        #Graphique prédiction
        train_predictions = sarima1.predict(start = train_data.index[0], end = train_data.index[-1])
        test_predictions = sarima1.predict(start = test_data.index[0], end = test_data.index[-1])
        
        fig=plt.figure(figsize=(10,6))
        plt.plot(train_data.index, train_data.values, label='Entrainement', color='red')
        plt.plot(test_data.index, test_data.values, label='Test', color='green')
        sarima1.predict(start=futureDate.index[0], end=futureDate.index[-1]).plot(label='Prédiction',color='blue');
        
        plt.xlim(train_data.index[0], futureDate.index[-1])
        #plt.ylim(min(train_data.min(), test_data.min()), max(train_data.max(),test_data.max()))
        
        plt.xlabel('Date')
        plt.ylabel('comptage')
        plt.title("Prédiction avec le modéle SARIMA",fontweight='bold')
        plt.xticks(rotation=45)
        plt.axvline(x= date(2023,3,15), color='black', linestyle='--')
        plt.axvline(x= date(2023,6,30), color='black', linestyle='--')
        plt.legend()
        plt.annotate('Entrainement',fontweight='bold',color='black', xy=(480, 100), xytext=(300, 100),xycoords='figure points')
        plt.annotate('Test',fontweight='bold',color='black', xy=(480, 100), xytext=(430, 100),xycoords='figure points')
        plt.annotate('Prédiction',fontweight='bold',color='black', xy=(480, 100), xytext=(540, 100),xycoords='figure points')
      
        if Prediction:
            
            st.pyplot(fig)
            
elif page == "Conclusion":
    st.write("### :blue[Conclusion]😀")
    # Ajoutez ici votre conclusion ou vos résultats finaux

