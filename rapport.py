import streamlit as st
import pandas as pd 
import numpy as np
import datetime as dt
#import matplotlib.pyplot as plt 
#import seaborn as sns
#from sklearn.preprocessing import StandardScaler
#from sklearn import preprocessing
#from sklearn.cluster import KMeans

st.markdown('Projet fil rouge DataScientest Promotion Data Analyst  Décembre 2021 – Juillet 2022')
#st.title ('ANALYSE DE SITE ECOMMERCE') 
st.markdown("<h1 style='text-align: center; color: blue;'>ANALYSE DE SITE ECOMMERCE</h1>", unsafe_allow_html=True)

from PIL import Image
image = Image.open('Image1.jpg')
st.image(image)

st.text('Participants :')
st.text('Thouaiba MALKI')
st.text('Dédé Synthia ATTAH')
st.text('Jawhar BEN HASSINE')

st.text('Mentor: Frédéric FRANCINE')


pages=['Le Pakistan en chiffres','Jeux de données & Data Cleaning','Data Viz','Pareto des Ventes', 'RFM Segmentation','Etude Kmeans', 'Etude détaillée par segment','Conclusion']
page=st.sidebar.radio("aller vers", pages)

df = pd.read_csv('Pakistan.csv', sep = ',', index_col='item_id')

#################################################################################################
#################################################################################################
################################################################################################
if page==pages[0]:
    st.header('Le Pakistan en chiffres')
    st.subheader('Le profil type d’un consommateur Pakistanais:')
    
    parag='''
    Le PIB en Pakistan représente 1 260 USD en 2020 . Sa population est en 
plein essor, 55,31 % du totale c’est les jeunes de 0 à 24 en 2020 avec un taux 
de croissance net de 1,99 % en 2021 .

Vue la taille très importante de sa population (plus de 216 millions de 
personnes, soit la cinquième population la plus nombreuse du monde - Banque 
mondiale, dernières données   disponibles),   le   Pakistan   représente   un 
marché   très   attrayant   de consommation.

Les consommateurs pakistanais deviennent de plus en plus sensibles aux marques.
Cette tendance crée une demande pour des produits qui n’étaient auparavant pas
connus   dans   le   pays,   en   particulier   dans   les   secteurs   de   
l’habillement   et   de l’équipement durable des foyers.

Malgré des augmentations de salaires, les consommateurs pakistanais continuent
de consacrer une part importante de leurs revenus à la satisfaction de besoins 
primaires tels que la nourriture (39,5% des dépenses totales de consommation) 
(Service de recherche   économique   du   Ministère   de   l’agriculture   des 
États-Unis,   dernières données disponibles).

En 2018, le Pakistan comptait 200,81 millions d'habitants, don’t 55,31 % c’est
des jeunes de 0 à 24 en 2020, 44,61% utilisaient Internet, soit un taux de 
pénétration de 22,2%, selon Internet World Stats.

En   raison   du   manque   de   ressources   et   d'infrastructures,   Le   
faible   niveau d'alphabétisation, les conditions économiques et la résistance 
culturelle rendent la pénétration de l'Internet limitée, et a considérablement 
diminué et l'accès à Internet mobile a augmenté, ce qui a entraîné une
pénétration globale plus élevée ; En mars 2017, le Pakistan comptait plus de 
137 millions d'utilisateurs de téléphones mobiles, 18 millions d'utilisateurs 
de téléphones intelligents en 2018.'''
    st.text(parag)
    st.subheader('Le marché du e-commerce:')
    parag1=''' 
    Le marché pakistanais du e-commerce devrait dépasser le milliard de dollars
d'ici fin 2018 en raison des nouveaux marchands de paiement en ligne et de la 
pénétration du haut débit (The Express Tribune).

Selon Statista, les ventes en ligne se sont élevées à 622 millions USD en 2017,
soit 0,34% des ventes au détail du pays. Cependant, la valeur réelle des ventes
en ligne est probablement beaucoup plus élevée, étant donné que la banque 
centrale du Pakistan n'a pris en compte que les paiements effectués par cartes 
de débit ou de crédit, qui ne représentaient qu'une petite partie de toutes 
les transactions.

Selon Euromonitor International, le nombre croissant de jeunes et de 
consommateurs urbains alimente le boom de la vente au détail sur Internet au 
Pakistan. Un marketing agressif et des droits de distribution exclusive avec 
des marques internationales ont été parmi les facteurs clés de cette croissance.

Le rapport annuel 2017 indiquait qu'il y avait 5 millions d'acheteurs en ligne 
dans le pays en 2017, 2 000 détaillants en ligne, dont environ 375 acceptaient 
les paiements numériques. En conséquence, plus de 90% des transactions étaient
payées en espèces à la livraison. Les smartphones sont le support commercial
 préféré des Pakistanais, au lieu des ordinateurs portables et des ordinateurs
de bureau. 

Daraz.pk a dominé les ventes au détail en ligne en 2016, avec une part de
marché de 13%; un Black Friday en ligne, qui a battu les records de vente en 
novembre 2015,Selon Daraz.pk, 76% de ce trafic provenait de téléphones mobiles 
en 2017. Les principaux acteurs de l'e-commerce dans le pays investissent donc 
énormément dans leurs applications et leurs sites Web compatibles avec les 
appareils mobiles. Les produits mobiles et électroniques constituent depuis 
quelques années la catégorie de produits la plus populaire parmi les acheteurs 
en ligne, qui se taillent la part du lion dans les ventes. Toutefois, en 2017, 
40% des ventes ont été générées par des clients ayant  commandé des produits  
de beauté,  des produits d'épicerie et  des articles ménagers.

Les principaux sites Web de commerce électronique au Pakistan sont olx.com.pk,
daraz.pk, pakwheels.com, zameen.com et kaymu.pk.

Alibaba   vient   tout   juste   d'entrer   sur   le   marché   pakistanais   
et  Amazon   cherche également à investir dans le pays.

Selon le   rapport   annuel   2017 publié   par   l'Autorité   des   
télécommunications du Pakistan, le commerce électronique est l'un des 
principaux moteurs du Pakistan numérique. Par conséquent, le gouvernement 
travaille à l'élaboration d'un cadre stratégique pour le e-commerce, afin de 
réglementer et de développer davantage le marché au Pakistan.

L’un des principaux problèmes qui a limité le développement du e-commerce au
Pakistan est l’absence d’options de paiement en ligne fiables ; l'entrée 
d'acteurs internationaux sur ce marché devrait résoudre cet obstacle de 
longue date.'''
    st.text(parag1)
    
    st.subheader('Contexte:')
    parag2='''
    Le projet a pour ambition l’analyse de données d’un site de vente de 
e-commerce en volume. Notre principal outil de travail repose sur l’un des 
plus grands ensembles de données de commandes de commerce d’électronique et de 
technologies de détail du Pakistan. De Mars 2016 à Août 2018, il contient 
environ un peu plus d’un million d’enregistrements de transactions.
Les données ont été collectées auprès de divers marchands de commerce 
électronique dans le cadre d’une étude de recherche. Il a pour vocation à 
connaître les besoins urgents sur le potentiel émergent du commerce 
électronique au Pakistan.
Variables:   The   dataset   contains   Item   ID,   Order   Status   
(Completed,   Cancelled,Refund), Date of Order, SKU, Price, Quantity, 
Grand Total, Category, Payment Method and Customer ID.  
Dès   lors   l’objectif   sera   d’étudier   ce   site   d’e-commerce   d’y   
appliquer   une segmentation client spécifique par la méthode RFM 
(Recency Frequency Monitary),une méthode récurrente pour un site de e-commerce.
L’enjeu a été donc pour nous de nettoyer et de comprendre les données du jeu au
regard de sa mauvaise qualité. Par la suite nous regarderons visuellement si 
les ventes de l’entreprise fonctionnent. Pour finir nous appliquerons des 
méthodes de clustering classiques et proposer des stratégies pour augmenter 
les ventes, fidéliser les clients,améliorer l’image du site.
Pour ce projet notre façon de travailler s’est découpé de la façon suivante :
1- Un travail individuel sur le dataset comme premier jet
2- Un point call hebdomadaire pour regrouper nos différents travaux
3- Recoupage des meilleurs informations pour le pre-processing  
4- Répartition des tâches.'''
    st.text(parag2)
    
    ####################################################################
    #######################################################################"
    ##############################################################
if page==pages[1]:
    st.header('Jeux de données & Data Cleaning') 
    st.subheader("Présentation:")
    
    parag0='''
Le dataset mis à notre disposition est un fichier csv. Il est disponible sur le
site   de   Kaggle.   Lien   pour   télécharger   le   dataset   :
https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-
dataset
Nous allons importer plusieurs librairies et ouvrir le dataset avec pandas
   
    '''
    st.text(parag0)
    
    ############## affichage du source
    code=''' df = pd.read_csv('C:/Users/Malki.T/Documents/workspace/00.run/Pakistan.csv', \nsep = ',', index_col='item_id')'''
    st.code(code, language='python')
      
    st.text('Affichage   des   infos   du   dataframe   pour   avoir   un   \npremier   aperçu des données ')
    ############ appercue du data frame
    st.dataframe(df.head(5))

    import io 
    buffer = io.StringIO() 
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    parag3=''' 
    On remarque que presque la moitier des données sont des lignes vides, donc c'est 
    important de les supprimer avant de commencer notre analyses.
    '''
    st.text(parag3)
############ suppression des lignes vides
    st.text("# suppression des lignes entirement vides")
    df.dropna(how='all', inplace= True)
    
    
    code1 = '''df.dropna(how='all', inplace= True)'''
    st.code(code1, language='python')
    
    parag4='''
    les données avec lesquelles nous allons travailler constituent un ensemble
    de données de commandes de commerce électronique de détail en provenance 
    du Pakistan. Il contient un demi-million d’enregistrements de transactions
    de juin 2016 à août 2018. Les données ont été recueillies dans le cadre 
    d’une étude de recherche
    '''
    st.text(parag4)
    parag5='''
    L’ensemble de données: L’ensemble de données contient des informations sur
    les détails de l’article, le mode d’expédition, le mode de paiement comme
    la carte de crédit, Easy-Paisa, Jazz-Cash, le paiement à la livraison, 
    les catégories de produits comme la mode, le mobile, l’électronique, 
    l’appareil, etc., la date de commande, le SKU, le prix, la quantité, 
    le total et l’ID client. 
    '''

    st.text(parag5)
    
    ############## suppréssion des colonnes inutiles
    st.subheader("suppression des colonnes :")
    
    df.drop(columns=['increment_id', 'sales_commission_code','discount_amount',
                     'BI Status',' MV ','Year', 'Month', 'FY','Unnamed: 21',
                     'Unnamed: 22','Unnamed: 23','Unnamed: 24','Unnamed: 25' ]
            ,axis=1,inplace=True)
    
    df.rename(columns={'sku': 'articles','grand_total':'amount','category_name_1': 'category_name','Working Date': 'Working_Date','Customer Since': 'Customer_Since','Customer ID': 'Customer_ID','M-Y': 'date'}, inplace=True)
    df['qty_ordered']=df['qty_ordered'].astype(int) 
    df.category_name= df.category_name.astype(str)
    
    st.text("On supprime les colonnes vides et colonnes inutiles" )
   
    code3='''
df.drop(columns=['increment_id', 'sales_commission_code','discount_amount',
'BI Status',' MV ','Year', 'Month', 'FY','Unnamed: 21','Unnamed: 22',
'Unnamed: 23','Unnamed: 24','Unnamed: 25' ],axis=1,inplace=True)"    
    '''
    st.code(code3, language='python')
    
  ################ colonne à garder et renommer les colonnes
    st.subheader('Variables: Colonnes à garder')
    
    parag6='''
 Les colonnes à garder sont  l’ID de l’article, l’état de la commande 
 (terminé, annulé, remboursé), la date de la commande, le SKU, le prix, la 
quantité, le total général, la catégorie, le mode de paiement et l’ID client.   
    '''
    st.text(parag6)
  
    
   
    st.subheader('Renommer les colonnes')    
     
    code4='''
df.rename(columns={'sku': 'articles','grand_total':'amount','category_name_1': 
'category_name','Working Date': 'Working_Date','Customer Since': 
'Customer_Since','Customer ID': 'Customer_ID','M-Y': 'date'}, inplace=True)   
    '''
    st.code(code4, language='python')             
   
   
    
    
         
    index_with_nan = df.index[df.isnull().any(axis=1)]
         
    df.drop(index_with_nan,0, inplace=True)
         
    df.isnull().sum() 

    st.subheader('Suppression des lignes contenant des NAs (210 lignes par rapport à plus de 500000):')
    
    parag7='''
Il existait 210 lignes avec des NAs, vue que le nombre des lignes présentant des
valeurs manquantes sont négligables par rapport à plus de 500000 lignes, nous 
avons opté le choix de supprimer ces lignes.    
    '''

    st.text(parag7)
    
    code19='''
     index_with_nan = df.index[df.isnull().any(axis=1)]
          
     df.drop(index_with_nan,0, inplace=True)
          
     df.isnull().sum() 
     
     '''
    st.code(code19, language='python')
    
    
    
    if st.checkbox('Afficher les valeurs manquantes'):
        st.dataframe(df.isnull().sum())
    
############################################################################################################################
############################################################
########################################################
import base64
#import time
if page==pages[2]:
    st.header('Data Viz')
    ################## intégrer
    
 #   st.subheader("voici un apperçue sur le rapport Power PI")
 #   src="C:/Users/Malki.T/Documents/workspace/01.rapport_ec/Présentation Projet e_commerce.pdf"
 #  with open(src,"rb") as f:
 #       base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
#    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    
 #   st.markdown(pdf_display, unsafe_allow_html=True)
    ############################ télécharger
    
    st.subheader("télécharger le Data Viz Power PI Rapport")
    
    
    with open("C:/Users/Malki.T/Documents/workspace/99.Archive/DataViz_PBI.pdf", "rb") as src:
        PDFbyte = src.read()

    st.download_button(label="Download Data Viz Power BI Rapport", 
        data=PDFbyte,
        file_name="DataViz_PBI.pdf",
        mime='application/octet-stream')
    
##############################################################################
#############################################################################################""""
#####################################################################################
if page==pages[3]:
    st.header('Pareto des Ventes')
             
    st.subheader("télécharger le Power PI Rapport")
    
    
    with open("C:/Users/Malki.T/Documents/workspace/99.Archive/Pareto_PBI.pdf", "rb") as src:
        PDFbyte = src.read()

    st.download_button(label="Download Pareto Power BI Rapport", 
        data=PDFbyte,
        file_name="Pareto_PBI.pdf",
        mime='application/octet-stream')
    
    st.subheader('Top Produits')
    
    with open("C:/Users/Malki.T/Documents/workspace/99.Archive/Top produits_PBI.pdf", "rb") as src:
        PDFbyte = src.read()

    st.download_button(label="Download Top Produits Power BI Rapport", 
        data=PDFbyte,
        file_name="Top produits_PBI.pdf",
        mime='application/octet-stream')
    
#########################################################################################################################
##################################################################################
##################################################################################

if page==pages[4]:
    st.header('RFM Segmentation') 
    st.subheader("idem avec Synthia")
    parag11='''
nous procederons à une segmentation de notre dataset: Nous allons faire une 
analyse RFM, pour cela nous allons extrère une dataset basée sur les dates, 
prix et quantité, nous aurons un tableau comme suit    
    '''
    st.text(parag11) 
    
    
    df_rfm= pd.read_csv("C:/Users/Malki.T/Documents/workspace/99.Archive/df_rfm_brute.csv", sep=";", index_col=0)
    st.dataframe(df_rfm.head(20))
    
    parag14='''
par la suite nous avons appliquer la fonction aggrégation pour calculer la 
recensy par rapport la dernière date de la data set "2018,8,28", pour calculer 
la fréquence de chaque client et le budget de chaque achat, et nous aurons le 
tableau suivant:    
    '''
    st.text(parag14)
    
    RFM_Table= pd.read_csv("C:/Users/Malki.T/Documents/workspace/99.Archive/df_rfm1.csv", sep=";", index_col=0)
    st.dataframe(RFM_Table.head(20))
    
    parag15='''
En se basant sur la dataframe RFM8Table, nous allons calculer le score de 
chaque recensy, frequency et monetary, et on aura le resultat suivant:    
    '''
    st.text(parag15)
         
    RFM_table= pd.read_csv("C:/Users/Malki.T/Documents/workspace/99.Archive/df_rfm2.csv", sep=";", index_col=0)
    st.dataframe(RFM_table.head(20))
    
    
    parag16='''
Notre objectif est d'affecter un score pour chaque client, et on aura le 
tableau suivant    
    '''
    st.text(parag16)
       
    RFM_Scor= pd.read_csv("C:/Users/Malki.T/Documents/workspace/99.Archive/df_rfm_score.csv", sep=";", index_col=0)
    st.dataframe(RFM_Scor.head(20))
    
    parag17='''
Par la suite, en se basant sur le score de chaque client, on peut classer les
client en différents segments comme vous pouvez le voir dans le tableau 
suivant:    
    '''
    st.text(parag17)  
        
    RFM_Score= pd.read_csv("C:/Users/Malki.T/Documents/workspace/99.Archive/df_rfm_customer_segmentation.csv", sep=";", index_col=0)
    st.dataframe(RFM_Score.head(20))
 ########################################         
    st.text("La distribution de segmantation Client:") 
    
    
    
    code16='''
plt.figure(figsize=(20,8))
plt.pie(RFM_table.Customer_segment.value_counts(),
        labels=RFM_table.Customer_segment.value_counts().index,
        autopct='%.0f%%')
plt.show()    
    
    st.code(code16, language='python')
    st.pyplot(fig7)   
    
    st.text('la distribution selon R,M & F')
    
    RFM_table.groupby('Customer_segment').agg({
    'recency' : ['mean', 'min','max'],
    'frequency' : ['mean', 'min','max'],
    'monetary' : ['mean','min','max','count']
    })
    
    fig8, ax = plt.subplots()
    plt.figure(figsize=(15,5))
    sns.barplot(x=RFM_table.Customer_segment, y=RFM_table.M_score);
    st.pyplot(fig8)
    
    fig9, ax = plt.subplots()
    plt.figure(figsize=(15,5))
    sns.barplot(x=RFM_table.Customer_segment, y=RFM_table.F_score);
    st.pyplot(fig9)
    
    fig10, ax = plt.subplots()
    plt.figure(figsize=(15,5))
    sns.barplot(x=RFM_table.Customer_segment, y=RFM_table.R_score);
    st.pyplot(fig10)
    
    parag12=''''''
La distribution des Customer segment est semblable par rapport à l'axe 
'monetory' et l'axe frequency, mais elle différente par rapport à l'axe 
'recency' : les segment de 'hight value custumer' et 'medium value custumer' 
ont une recensy score faible.    
    '''
    
    
####################################################################################"
####################################################################################"
####################################################################################
   
if page==pages[5]:
    st.header('Etude Kmeans') 
    
    st.text("nous important les bibli nécessaires")
    
    code17='''
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans    
    '''
    st.code(code17, language='python')
    
    code18='''
    temp1=[ 'recency', 'frequency', 'monetary']
    RFM_kmean=RFM_table[temp1]
    RFM_kmean.head()
    
    scaler = StandardScaler()
    x_scaled=scaler.fit(RFM_kmean)
    x_scaled = scaler.fit_transform(RFM_kmean)
    x_scaled
    
    x_scaled=pd.DataFrame(x_scaled, columns = RFM_kmean.columns)
    x_scaled.head(30)
    
    model=KMeans(n_clusters=(2)).fit(x_scaled)

    fig=plt.figure(figsize=(20,20))
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(x_scaled['recency'],x_scaled['frequency'], x_scaled['monetary'],
          cmap='brg',c=model.predict(x_scaled))
    ax.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,0],c='black');
    '''
    st.code(code18, language='python')
    
    
    src1="C:/Users/Malki.T/Documents/workspace/99.Archive/kmeans_graph.pdf"
    with open(src1,"rb") as f1:
        base64_pdf = base64.b64encode(f1.read()).decode('utf-8')
        
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    
    st.markdown(pdf_display, unsafe_allow_html=True)
    
##########################################################################################


#    df['created_at']= pd.to_datetime(df['created_at'])
#   df['Working_Date']= pd.to_datetime(df['Working_Date'])
#    df['Customer_Since']= pd.to_datetime(df['Customer_Since'])
#    df['date']= pd.to_datetime(df['date'])

#    df['Year'] = df['created_at'].dt.year
#    df['Month'] = df['created_at'].dt.month
#    df['Day']  = df['created_at'].dt.day
    
#    temp=['Customer_ID','price', 'amount', 'date', 'qty_ordered']
#    RFM_data=df[temp]
    
    
#    NOW = dt.datetime(2018,8,28)

    # RFM Table
#    RFM_table=RFM_data.groupby('Customer_ID').agg({'date': lambda x: (NOW - x.max()).days, # Recency
#                                                'qty_ordered': lambda x: len(x.unique()), # Frequency
#                                                'amount': lambda x: x.sum()})    # Monetary 
    

#    RFM_table['date'] = RFM_table['date'].astype(int)

#    RFM_table.rename(columns={'date': 'recency', 
#                         'qty_ordered': 'frequency',
#                         'amount': 'monetary'}, inplace=True)
#    RFM_table.head()
    
        
#    st.dataframe(RFM_table.head())

    
#    RFM_table['R_score'] = RFM_table['recency'].rank(ascending=False)
#    RFM_table['F_score'] = RFM_table['frequency'].rank(ascending=True)
#    RFM_table['M_score'] = RFM_table['monetary'].rank(ascending=True)
     
    # normalizing the rank of the customers
#    RFM_table['R_norm'] = (RFM_table['R_score']/RFM_table['R_score'].max())*100
#    RFM_table['F_norm'] = (RFM_table['F_score']/RFM_table['F_score'].max())*100
#    RFM_table['M_norm'] = (RFM_table['F_score']/RFM_table['M_score'].max())*100
    
    
 
#    RFM_table.drop(columns=['R_score', 'F_score', 'M_score'])
#    RFM_table = RFM_table.reset_index(level=0)
#    RFM_table.head()
    
    #st.dataframe(RFM_data.head(20))
    
#    RFM_table['RFM_Score'] = 0.15*RFM_table['R_norm']+0.28 * \
#    RFM_table['F_norm']+0.57*RFM_table['M_norm']
#    RFM_table['RFM_Score'] *= 0.05
#    RFM_table = RFM_table.round(2)
#    RFM_table[['Customer_ID', 'RFM_Score']].head(7)
    
#    RFM_table["Customer_segment"] = np.where(RFM_table['RFM_Score'] > 4.5, "Top Customers",
#                                      (np.where(RFM_table['RFM_Score'] > 4,"High value Customer",
#                                    (np.where(RFM_table['RFM_Score'] > 3, "Medium Value Customer",
#                                    np.where(RFM_table['RFM_Score'] > 1.6,'Low Value Customers', 
#                                             'Lost Customers'))))))
#    RFM_table[['Customer_ID', 'RFM_Score', 'Customer_segment']].head(20)
   
    #st.dataframe(RFM_table[['Customer_ID', 'RFM_Score', 'Customer_segment']].head(10))'''
    
    
  ####################################################################################
#######################################################################################
##########################################################################################  
if page==pages[6]:
    st.header('Etude détaillée des articles achetés par client')
    
    parag13='''
Dans ce que suit, on va extraire chauque segment de client dans une data frame,
pour faciliter l'étude du comportement de chaque segment du client selon 
le moyen de payement et la catégorie des produits achétés :
    
   '''
    
    '''
    best_df = pd.DataFrame()
    best_df["Customer_ID"] = RFM_table[RFM_table["Customer_segment"] == "Top Customers"].index
    best_df.head(5)
    #best_df.to_excel("best_customers.xlsx", sheet_name='Best Customers Index')
    
    good_df = pd.DataFrame()
    good_df["Customer_ID"] = RFM_table[RFM_table["Customer_segment"] == "High value Customer"].index
    good_df.head()
    #good_df.to_excel("good_customers.xlsx", sheet_name='Good Customers Index')
    
    medium_df = pd.DataFrame()
    medium_df["Customer_ID"] = RFM_table[RFM_table["Customer_segment"] == "Medium Value Customer"].index
    medium_df.head()
    #medium_df.to_excel("medium_customers.xlsx", sheet_name='Medium Customers Index')
    
    worst_df = pd.DataFrame()
    worst_df["Customer_ID"] = RFM_table[RFM_table["Customer_segment"] == "Low Value Customers"].index
    worst_df.head()
    #worst_df.to_excel("worst_customers.xlsx", sheet_name='Worst Customers Index')
    
    st.subheader('Best Customers-Moyens de payement')
    
    temp=[ 'Customer_ID','articles', 'category_name', 'status','date','payment_method']
    df_select=df[temp]
    df_select = df_select.loc[df_select['status'] =='complete']
    merge_best=best_df.merge(df_select, how='left', on='Customer_ID')
    merge_best.shape
    
    index_with_nan = merge_best.index[merge_best.isnull().any(axis=1)]
    merge_best.drop(index_with_nan,0, inplace=True)
    merge_best.isnull().sum()
    
    merge_best.shape
    best_df.head(10)
    
    import itertools as it
    from collections import Counter

    count=Counter()

    for row in merge_best['articles']:
        row_list=row.split(',')
        count.update(Counter(it.combinations(row_list,1)))
        
    count.most_common(20)
    merge_best[merge_best['articles']=='kcc_krone deal']
    
    print(len(merge_best[merge_best['payment_method']=='cod'])/len(merge_best)+len(merge_best[merge_best['payment_method']=='cashatdoorstep'])/len(merge_best) )
    cash_best=(len(merge_best[merge_best['payment_method']=='cod'])/len(merge_best)+len(merge_best[merge_best['payment_method']=='cashatdoorstep'])/len(merge_best) )*100 
    
    marketing_best=(len(merge_best[merge_best['payment_method']=='jazzvoucher'])/len(merge_best) +len(merge_best[merge_best['payment_method']=='marketingexpense'])/len(merge_best) )*100

    credit_best=(len(merge_best[merge_best['payment_method']=='customercredit'])/len(merge_best) +
                 len(merge_best[merge_best['payment_method']=='productcredit'])/len(merge_best) +
                 len(merge_best[merge_best['payment_method']=='financesettlement'])/len(merge_best) )*100

    card_best=100-(cash_best+marketing_best+credit_best)

    st.subheader('Best Customers-catégories')
    
    electro_best=(len(merge_best[merge_best['category_name']=='Mobiles & Tablets'])/len(merge_best)+
     len(merge_best[merge_best['category_name']=="Computing"])/len(merge_best)+
     len(merge_best[merge_best['category_name']=="Appliances"])/len(merge_best))*100

    fashion_best=(len(merge_best[merge_best['category_name']=="Women's Fashion"])/len(merge_best)+
     len(merge_best[merge_best['category_name']=="Men's Fashion"])/len(merge_best)+
     len(merge_best[merge_best['category_name']=="Kids & Baby"])/len(merge_best))*100
    
    epicery_best=(len(merge_best[merge_best['category_name']=="Soghaat"])/len(merge_best)+
     len(merge_best[merge_best['category_name']=="Superstore"])/len(merge_best)+
     len(merge_best[merge_best['category_name']=="Beauty & Grooming"])/len(merge_best))*100
    
    home_best=(len(merge_best[merge_best['category_name']=="Home & Living"])/len(merge_best))*100
    
    entertainement_best=(len(merge_best[merge_best['category_name']=='Entertainment'])/len(merge_best)+
     len(merge_best[merge_best['category_name']=="Health & Sports"])/len(merge_best))*100
    
    others_best=100-(electro_best+fashion_best+epicery_best+home_best+entertainement_best)
    
    st.subheader('Good Customer: moyen de payement')
    merge_good=good_df.merge(df_select, how='left', on='Customer_ID')
    index_with_nan = merge_good.index[merge_good.isnull().any(axis=1)]
    merge_good.drop(index_with_nan,0, inplace=True)
    merge_good.isnull().sum()
    
    cash_good=(len(merge_good[merge_good['payment_method']=='cod'])/len(merge_good)+
len(merge_good[merge_good['payment_method']=='cashatdoorstep'])/len(merge_good))*100
    
    marketing_good=(len(merge_good[merge_good['payment_method']=='jazzvoucher'])/len(merge_good) +
len(merge_good[merge_good['payment_method']=='marketingexpense'])/len(merge_good) )*100
    
    credit_good=(len(merge_good[merge_good['payment_method']=='customercredit'])/len(merge_good) +
len(merge_good[merge_good['payment_method']=='productcredit'])/len(merge_good) +
len(merge_good[merge_good['payment_method']=='financesettlement'])/len(merge_good))*100
    
    card_good=100-(cash_good+marketing_good+credit_good)
    
    electro_good=(len(merge_good[merge_good['category_name']=='Mobiles & Tablets'])/len(merge_good)+
     len(merge_good[merge_good['category_name']=="Computing"])/len(merge_good)+
     len(merge_good[merge_good['category_name']=="Appliances"])/len(merge_good))*100
    
    fashion_good=(len(merge_good[merge_good['category_name']=="Women's Fashion"])/len(merge_good)+
     len(merge_good[merge_good['category_name']=="Men's Fashion"])/len(merge_good)+
     len(merge_good[merge_good['category_name']=="Kids & Baby"])/len(merge_good))*100
    
    epicery_good=(len(merge_good[merge_good['category_name']=="Soghaat"])/len(merge_good)+
     len(merge_good[merge_good['category_name']=="Superstore"])/len(merge_good)+
     len(merge_good[merge_good['category_name']=="Beauty & Grooming"])/len(merge_good))*100
    
    home_good=(len(merge_good[merge_good['category_name']=="Home & Living"])/len(merge_good))*100
    
    entertainement_good=(len(merge_good[merge_good['category_name']=='Entertainment'])/len(merge_good)+
     len(merge_good[merge_good['category_name']=="Health & Sports"])/len(merge_good))*100
    
    others_good=100-(electro_good+fashion_good+epicery_good+home_good+entertainement_good)
    
    # Medium Costomer
    
    merge_medium=medium_df.merge(df_select, how='left', on='Customer_ID')
    
    index_with_nan = merge_medium.index[merge_medium.isnull().any(axis=1)]
    merge_medium.drop(index_with_nan,0, inplace=True)
    merge_medium.isnull().sum()

    cash_medium=(len(merge_medium[merge_medium['payment_method']=='cod'])/len(merge_medium)+
len(merge_medium[merge_medium['payment_method']=='cashatdoorstep'])/len(merge_medium))*100
    
    marketing_medium=(len(merge_medium[merge_medium['payment_method']=='jazzvoucher'])/len(merge_medium) +
len(merge_medium[merge_medium['payment_method']=='marketingexpense'])/len(merge_medium) )*100
    
    credit_medium=(len(merge_medium[merge_medium['payment_method']=='customercredit'])/len(merge_medium)+
len(merge_medium[merge_medium['payment_method']=='productcredit'])/len(merge_medium)+
len(merge_medium[merge_medium['payment_method']=='financesettlement'])/len(merge_medium))*100
    
    card_medium=100-(cash_medium+marketing_medium+credit_medium)
    
    #catégories
    electro_medium=(len(merge_medium[merge_medium['category_name']=='Mobiles & Tablets'])/len(merge_medium)+
     len(merge_medium[merge_medium['category_name']=="Computing"])/len(merge_medium)+
     len(merge_medium[merge_medium['category_name']=="Appliances"])/len(merge_medium))*100
    
    fashion_medium=(len(merge_medium[merge_medium['category_name']=="Women's Fashion"])/len(merge_medium)+
     len(merge_medium[merge_medium['category_name']=="Men's Fashion"])/len(merge_medium)+
     len(merge_medium[merge_medium['category_name']=="Kids & Baby"])/len(merge_medium))*100
    
    epicery_medium=(len(merge_medium[merge_medium['category_name']=="Soghaat"])/len(merge_medium)+
     len(merge_medium[merge_medium['category_name']=="Superstore"])/len(merge_medium)+
     len(merge_medium[merge_medium['category_name']=="Beauty & Grooming"])/len(merge_medium))*100
    
    home_medium=(len(merge_medium[merge_medium['category_name']=="Home & Living"])/len(merge_medium))*100
    
    entertainement_medium=(len(merge_medium[merge_medium['category_name']=='Entertainment'])/len(merge_medium)+
     len(merge_medium[merge_medium['category_name']=="Health & Sports"])/len(merge_medium))*100
    
    others_medium=100-(electro_medium+fashion_medium+epicery_medium+home_medium+entertainement_medium)
    
    #Worst costumer
    
    merge_worst=worst_df.merge(df_select, how='left', on='Customer_ID')
    
    index_with_nan = merge_worst.index[merge_worst.isnull().any(axis=1)]
    merge_worst.drop(index_with_nan,0, inplace=True)
    merge_worst.isnull().sum()
    
    cash_worst=(len(merge_worst[merge_worst['payment_method']=='cod'])/len(merge_worst)+
len(merge_worst[merge_worst['payment_method']=='cashatdoorstep'])/len(merge_worst))*100
    
    marketing_worst=(len(merge_worst[merge_worst['payment_method']=='jazzvoucher'])/len(merge_worst) +
len(merge_worst[merge_worst['payment_method']=='marketingexpense'])/len(merge_worst) )*100
    
    credit_worst=(len(merge_worst[merge_worst['payment_method']=='customercredit'])/len(merge_worst)+
len(merge_worst[merge_worst['payment_method']=='productcredit'])/len(merge_worst)+
len(merge_worst[merge_worst['payment_method']=='financesettlement'])/len(merge_worst))*100
    
    card_worst=100-(cash_worst+marketing_worst+credit_worst)
    
    electro_worst=(len(merge_worst[merge_worst['category_name']=='Mobiles & Tablets'])/len(merge_worst)+
     len(merge_worst[merge_worst['category_name']=="Computing"])/len(merge_worst)+
     len(merge_worst[merge_worst['category_name']=="Appliances"])/len(merge_worst))*100
    
    fashion_worst=(len(merge_worst[merge_worst['category_name']=="Women's Fashion"])/len(merge_worst)+
     len(merge_worst[merge_worst['category_name']=="Men's Fashion"])/len(merge_worst)+
     len(merge_worst[merge_worst['category_name']=="Kids & Baby"])/len(merge_worst))*100
    
    epicery_worst=(len(merge_worst[merge_worst['category_name']=="Soghaat"])/len(merge_worst)+
     len(merge_worst[merge_worst['category_name']=="Superstore"])/len(merge_worst)+
     len(merge_worst[merge_worst['category_name']=="Beauty & Grooming"])/len(merge_worst))*100
    
    home_worst=(len(merge_worst[merge_worst['category_name']=="Home & Living"])/len(merge_worst))*100
    
    entertainement_worst=(len(merge_worst[merge_worst['category_name']=='Entertainment'])/len(merge_worst)+
     len(merge_worst[merge_worst['category_name']=="Health & Sports"])/len(merge_worst))*100
    
    others_worst=100-(electro_worst+fashion_worst+epicery_worst+home_worst+entertainement_worst)
    
    moy_electro=(electro_best+electro_good+electro_medium+electro_worst)/4
    moy_fashion=(fashion_best+fashion_good+fashion_medium+fashion_worst)/4
    moy_epicery=(epicery_best+epicery_good+epicery_medium+epicery_worst)/4
    moy_home=(home_best+home_good+home_medium+home_worst)/4
    moy_entertainement=(entertainement_best+entertainement_good+entertainement_medium+entertainement_worst)/4
    moy_others=(others_best+others_good+others_medium+others_worst)/4
    
    
    data_categories = {'Electronics':{'best':electro_best,'good':electro_good,'medium':electro_medium,'worst':electro_worst,'moyenne':moy_electro},
        'Fashions':{'best':fashion_best,'good':fashion_good,'medium':fashion_medium,'worst':fashion_worst,'moyenne':moy_fashion},
         'Stores':{'best':epicery_best,'good':epicery_good,'medium':epicery_medium,'worst':epicery_worst,'moyenne':moy_epicery},
        'home':{'best':home_best,'good':home_good,'medium':home_medium, 'worst':home_worst,'moyenne':moy_home},
'Entertainements':{'best':entertainement_best,'good':entertainement_good,'medium':entertainement_medium, 'worst':entertainement_worst,'moyenne':moy_entertainement},
        'Others':{'best':others_best,'good':others_good,'medium':others_medium, 'worst':others_worst,'moyenne':moy_others}
       }
    
    
    
    stat_categories = pd.DataFrame(data=data_categories)
    
    
    moy_cash=(cash_best+cash_good+cash_medium+cash_worst)/4
    moy_marketing=(marketing_best+marketing_good+marketing_medium+marketing_worst)/4
    moy_credit=(credit_best+credit_good+credit_medium+credit_worst)/4
    moy_card=(card_best+card_good+card_medium+card_worst)/4
    
    
    data_payment_meth = {'cash':{'best':cash_best,'good':cash_good,'medium':cash_medium,'worst':cash_worst,'moyenne':moy_cash},
        'marketing':{'best':marketing_best,'good':marketing_good,'medium':marketing_medium,'worst':marketing_worst,'moyenne':moy_marketing},
         'credit':{'best':credit_best,'good':credit_good,'medium':credit_medium,'worst':credit_worst,'moyenne':moy_credit},
        'credit cart & internet':{'best':card_best,'good':card_good,'medium':card_medium, 'worst':card_worst,'moyenne':moy_card}
       }
    
    
    stat_paymen_methode = pd.DataFrame(data=data_payment_meth)
    
    st.subheader('stats moyen de payement')
    stat_paymen_methode.head()
    st.dataframe(stat_paymen_methode.head())
    st.subheader('stats catégories')
    stat_categories.head()
    st.dataframe(stat_categories.head())
   '''
    

    '''
    En se basant sur le code qui se trouve en haut, nous avons pu résumé 
    quelques statistiques dans les tableaux suivants:
    '''
    st.subheader('Statistiques selon les moyens de payement sur le site')
    stats_moyens_payements= pd.read_csv("C:/Users/Malki.T/Documents/workspace/99.Archive/stats_myens_payement.csv", sep=";", index_col=0)
    st.dataframe(stats_moyens_payements.head(7))
    
    
    st.subheader("Statistiques selon les catégories d'achat des pakistanais")
    stats_category= pd.read_csv("C:/Users/Malki.T/Documents/workspace/99.Archive/stas_categry.csv", sep=";", index_col=0)
    st.dataframe(stats_category.head(7))
    
    st.subheader("télécharger le Power PI Rapport")
    
    
    with open("C:/Users/Malki.T/Documents/workspace/99.Archive/RFM_PBI.pdf", "rb") as src:
        PDFbyte = src.read()

    st.download_button(label="Download RFM Power BI Rapport", 
        data=PDFbyte,
        file_name="RFM_PBI.pdf",
        mime='application/octet-stream')
#######################################################################################
#######################################################################################
########################################################################################


if page==pages[7]:
    st.header('Conclusion')

    '''
    our conclure, d’un de vue globale, en volume les clients achètent 
principalement des produits des catégories suivantes :
    
Electronics : dans nos sociétés modernes, cette catégorie représente un besion
quasi-primaire car elle nous connecte au monde, à l’information et aux achats, 
et surtout les achats des smartphones qui sont en tetes et qui sont surtout 
utilisés par les jeunes.


Fashion 30% : il s’agit d’un besoin de se vêtir et participe à la valorisation de 
soi, et les vetements de marque se sont les plus demandé.


Home utility : en proportion, ils représentent tous les achats d’électroménagers
relatif aux fonctionnement d’une maison.

La nourritures représente aussi un marché très porteurs (30%)vues le nombres
important des fétes nationales liées aux différentes religions et croyances
et vue la position géographique du pakistan

On remarque que notre études se rapprochent des statistiques menés sur la 
population pakistanaise selon le comportement d'acaht d'un client lamda.

Concernant les modes de payements, il trouve que les pakistanais payent 
essentiellement en cash 70%, une bonne partie en crédit 27% et seulement
2% en ligne (ces ahat se sont concentrés surtout dans les grandes villes
et les statistiques auront une tendance de hausse dans les années qui viennent
vue que les stratégies politiques actuellent encouragent et facilitent le 
le developpement economiques des autres régions), l'entrée d'acteurs 
internationaux sur ce marché devrait résoudre cet obstacle de longue date 
sauf que le climat géoplolitique n'est pas très encourageant.


D’un point de vue micro, on observe une segmentation en six catégories, dont 
deux sont des clients fidèles (commerces de détail, et grossîtes) et les 
loyaux qui ont une certaine régulater d’achat.


Il serait intéressant de se positionner sur   les clients prometteurs afin de 
les faire basculer dans une des catégories cités plus haut. Il pourrait 
correspondre à des étudiants primo-arrivant sur le marchés du travail. 

Mettre en place une récupération d’adresses email, afin de les tenir informer 
des nouveautés, des réductions, des bon plans, avec comme attente une 
meilleure fidélisation.

Concernant les clients à risque, leur comportement se traduit par de faibles 
quantités achetés en plus de produits peu cher avec des achats récents. 
Il faudrait pour ne pas les perdre se cibler leur dernier achats et leur 
proposer des promotions. 

Par rapport aux clients qui hiberne, c’est-à-dire que leur comportement ne 
saute pas aux yeux, leur comportements étant similaire à ceux qui sont à 
risque, il est préférable de appliquer la même stratégie.



    '''
    
    st.markdown("<h1 style='text-align: center; color: blue;'>MERCI</h1>", unsafe_allow_html=True)
    
    
