import streamlit as st
import pandas as pd
import base64


st.markdown("<h8 style='text-align: left; color: red;'>Projet fil rouge DataScientest Promotion Data Analyst  Décembre 2021 – Juillet 2022</h1>", unsafe_allow_html=True)
st.markdown('')
st.markdown("<h1 style='text-align: center; color: blue;'>ANALYSE DE SITE ECOMMERCE</h1>", unsafe_allow_html=True)

from PIL import Image
image = Image.open('Image1.jpg')
st.image(image,width=800) 

st.markdown("<h6 style='text-align: left; color: blue;'>Participants :</h1>", unsafe_allow_html=True)
st.text('Thouaiba MALKI'   '//'  'Dédé Synthia ATTAH'     '// '  'Jawhar BEN HASSINE')
st.markdown("<h6 style='text-align: left; color: blue;'>Mentor:</h1>", unsafe_allow_html=True)
st.text('Frédéric FRANCINE')


pages=['Le Pakistan en chiffres','Jeux de données & Data Cleaning','Data Viz','Pareto des Ventes', 'RFM Segmentation et Etude Kmeans', 'Etude détaillée par segment','Conclusion']
page=st.sidebar.radio("aller vers", pages)

df = pd.read_csv('Pakistan.zip', sep = ',', index_col='item_id')






































if page==pages[0]:
    #st.header('')
    st.markdown("<h3 style='text-align: left; color: blue;'>Le Pakistan en chiffres</h1>", unsafe_allow_html=True)

    st.subheader("\n  \n  \n \n  \n  \n") 
    
    
    
    tex1="1 260 $ en 2020"
    tex2="44,61%"
    tex3="0,34% des ventes"
    tex4="90%"
    tex5="55,31%"
    tex6="622 m $ en 2017"
    tex7="5 millions"
    tex8="76%"
    tex9="1,99 % en 2021"
    tex10="39,5% des dépenses"
    tex11="2000"
    tex12="40%"
    
    col1, col2, col3= st.columns(3)
        
    col1.subheader("PIB")
    col1.write(tex1)
    col1.subheader("Utilisation \ninternet")
    col1.write(tex2)
    col1.subheader("Paiements\carte")
    col1.write(tex3)
    col1.subheader("Transactions en espèces")
    col1.write(tex4)


    col2.subheader("% Jeune")
    col2.write(tex5)
    col2.subheader("Ventes en ligne")
    col2.write(tex6)
    col2.subheader("Acheteurs en ligne")
    col2.write(tex7)
    col2.subheader("Téléphones mobiles")
    col2.write(tex8)


    col3.subheader("Taux de \ncroissance net")
    col3.write(tex9)
    col3.subheader("Nourritures")
    col3.write(tex10)
    col3.subheader("Détaillants \nen ligne")
    col3.write(tex11)
    col3.subheader("Produits de beauté")
    col3.write(tex12)


    
 







































if page==pages[1]:
    st.header('Jeux de données & Data Cleaning') 
    st.subheader("Présentation:")
    
    ############## affichage du source
    code=''' df = pd.read_csv('Pakistan.csv', \nsep = ',', index_col='item_id')'''
    st.code(code, language='python')
      
    st.dataframe(df.head(5))

    import io 
    buffer = io.StringIO() 
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

   
############ suppression des lignes vides
    st.subheader("Suppression des lignes entirement vides")
    df.dropna(how='all', inplace= True)
    
    
    code1 = '''df.dropna(how='all', inplace= True)'''
    st.code(code1, language='python')

    ############## suppréssion des colonnes inutiles
    st.subheader("Suppression des colonnes inutiles :")
    
    df.drop(columns=['increment_id', 'sales_commission_code','discount_amount',
                     'BI Status',' MV ','Year', 'Month', 'FY','Unnamed: 21',
                     'Unnamed: 22','Unnamed: 23','Unnamed: 24','Unnamed: 25' ]
            ,axis=1,inplace=True)
    
    df.rename(columns={'sku': 'articles','grand_total':'amount','category_name_1': 'category_name','Working Date': 'Working_Date','Customer Since': 'Customer_Since','Customer ID': 'Customer_ID','M-Y': 'date'}, inplace=True)
    df['qty_ordered']=df['qty_ordered'].astype(int) 
    df.category_name= df.category_name.astype(str)

    code3='''
df.drop(columns=['increment_id', 'sales_commission_code','discount_amount',
'BI Status',' MV ','Year', 'Month', 'FY','Unnamed: 21','Unnamed: 22',
'Unnamed: 23','Unnamed: 24','Unnamed: 25' ],axis=1,inplace=True)"    
    '''
    st.code(code3, language='python')
    
  ################ colonne à garder et renommer les colonnes
   

    st.subheader('Renommer les colonnes à gaerder')    
     
    code4='''
df.rename(columns={'sku': 'articles','grand_total':'amount','category_name_1': 
'category_name','Working Date': 'Working_Date','Customer Since': 
'Customer_Since','Customer ID': 'Customer_ID','M-Y': 'date'}, inplace=True)   
    '''
    st.code(code4, language='python')             
   
         
    index_with_nan = df.index[df.isnull().any(axis=1)]
         
    df.drop(index_with_nan,0, inplace=True)
         
    df.isnull().sum() 

    st.markdown("<h6 style='text-align: left; color: blue;'>Vérifions notre base :</h1>", unsafe_allow_html=True)  
        
    if st.checkbox('Afficher les valeurs manquantes'):
        st.dataframe(df.isnull().sum()) 
    
    st.markdown("<h3 style='text-align: left; color: blue;'>Notre base est prête :)</h1>", unsafe_allow_html=True)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if page==pages[2]: 
    st.header('Data Viz')
    
    from PIL import Image
    image6 = Image.open('dataviz.jpg')
    st.image(image6, width=600)
    
       
    st.subheader("Télécharger le Data Viz Power BI Rapport")
    
    
    with open("DataViz_PBI.pdf", "rb") as src:
        PDFbyte = src.read()

    st.download_button(label="Download Data Viz Power BI Rapport", 
        data=PDFbyte, 
        file_name="DataViz_PBI.pdf",
        mime='application/octet-stream')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if page==pages[3]:
    st.header('Pareto des Ventes et Top 1000 Produits')
             
    st.subheader("télécharger le Power BI Rapport")
    
    from PIL import Image
    image6 = Image.open('pareto.jpg')
    st.image(image6, width=600)
        
    
    
    with open("Pareto_PBI.pdf", "rb") as src:
        PDFbyte = src.read()

    st.download_button(label="Download Pareto Power BI Rapport", 
        data=PDFbyte,
        file_name="Pareto_PBI.pdf",
        mime='application/octet-stream')  
    
    st.subheader('Top Produits')
    
    with open("Top produits_PBI.pdf", "rb") as src:
        PDFbyte = src.read()

    st.download_button(label="Download Top Produits Power BI Rapport", 
        data=PDFbyte,
        file_name="Top produits_PBI.pdf",
        mime='application/octet-stream')
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

if page==pages[4]:
    st.header('RFM Segmentation et Etude Kmeans') 
    
    st.markdown("<h6 style='text-align: left; color: blue;'>Extraction des colonnes cocernées</h1>", unsafe_allow_html=True) 
    
    df_rfm= pd.read_csv("df_rfm_brute.csv", sep=";", index_col=0)
    st.dataframe(df_rfm.head(2))
    
    st.markdown("<h6 style='text-align: left; color: blue;'>Calcul de la récence, fréquence et montant</h1>", unsafe_allow_html=True) 
    
    RFM_Table= pd.read_csv("df_rfm1.csv", sep=";", index_col=0)
    st.dataframe(RFM_Table.head(2))
    
    st.markdown("<h6 style='text-align: left; color: blue;'>Score des récences, fréquence et montant</h1>", unsafe_allow_html=True) 
         
    from PIL import Image
    image3 = Image.open('rfm.jpg')
    st.image(image3,width=800) 
    
    RFM_table= pd.read_csv("df_rfm2.csv", sep=";", index_col=0)
    st.dataframe(RFM_table.head(2))
    
    st.markdown("<h6 style='text-align: left; color: blue;'>Calcul du score RFM pour chaque client</h1>", unsafe_allow_html=True)
    
    
       
    RFM_Scor= pd.read_csv("df_rfm_score.csv", sep=";", index_col=0)
    st.dataframe(RFM_Scor.head(2))
    
    st.markdown("<h6 style='text-align: left; color: blue;'>Désignation segmentation client</h1>", unsafe_allow_html=True) 
    
    RFM_Score= pd.read_csv("df_rfm_customer_segmentation.csv", sep=";", index_col=0)
    st.dataframe(RFM_Score.head(2))
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
    
    
    from PIL import Image
    image2 = Image.open('imgkmeans.jpg')
    st.image(image2,width=800) 


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
if page==pages[5]:
    st.header('Etude détaillée des articles achetés par Segment')
    
    from PIL import Image
    image5 = Image.open('loupejpg.jpg')
    st.image(image5) 
    
    st.subheader("télécharger le Power BI Rapport")
    
    
    with open("RFM_PBI.pdf", "rb") as src:
        PDFbyte = src.read()

    st.download_button(label="Download RFM Power BI Rapport",  
        data=PDFbyte,
        file_name="RFM_PBI.pdf",
        mime='application/octet-stream')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

if page==pages[6]:
    st.header('Conclusion')

    tex1="le type de clients de ce segment sont des commerçants, donc on peut utiliser les objets publicitaires pour les remercier via des cadeaux d’affaires.…"
    tex3="Se sont des particuliers, on peut récompenser les meilleurs clients, influenceurs, journalistes… "
    tex2="Puisque c'est le segment qui utilise le plus le payment marketing, on peut utiliser les cadeaux publicitaires pour marquer l’esprit (lancement de produits, prise de contact…)"
    tex4="C'est le segment qui a le plus besoin de l'animation, donc on peut l'animer sur les réseaux sociaux..."
    
    
    
    col1, col2= st.columns(2)
        
    col1.subheader("Segment VIP")
    col1.write(tex1)
    col1.subheader("Segment GOLD")
    col1.write(tex3)
    


    col2.subheader("Segment SILVER")
    col2.write(tex2)
    col2.subheader("Segment BRONZE")
    col2.write(tex4)
    
    '''
    De point de vue générale, on peut concidérer que l'objectif commun entre 
les qutres segments sera l'augentation de la fréquentation (visiter plus pour 
acheter plus)..
    ''' 
    
    st.subheader("Power BI")   
    
    with open("Power_BI.pdf", "rb") as src:
        PDFbyte = src.read()

    st.download_button(label="Power BI",  
        data=PDFbyte,
        file_name="Power_BI.pdf",
        mime='application/octet-stream')
    
    st.markdown("<h1 style='text-align: center; color: blue;'>MERCI !!!</h1>", unsafe_allow_html=True)





