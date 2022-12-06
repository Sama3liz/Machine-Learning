import re  
import nltk
import numpy as np
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import textdistance

loaded_model = pickle.load(open("static/bin/modelo1", 'rb'))  # abrir modelo 
loaded_tf_idf = pickle.load(open("static/bin/tf_idf", 'rb'))  # Con esto abres el modelo las 2 lineas 
index_labels = ['Modelo 1','Modelo 2','Modelo 3','Modelo 4','Modelo 5']

def nlp(text):
    text = re.sub('[^A-Za-z\u00C0-\u017F]+',' ',str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    diccionario = [word for word in tokens if not word in stopwords.words("spanish")]
    return diccionario
    
def tipoClase(pred):
    if pred == 0:
        clase = "Insultos Sexuales (clase 1)"
    if pred == 1:
        clase = "Insultos Discriminatorios (clase 2)"
    if pred == 2:
        clase = "Insultos Xenofobias (clase 3)"
    if pred == 3:
        clase = "Insultos por casta (clase 4)"
    if pred == 4:
        clase = "Insultos internacionales (clase 5)"
    if pred == 5:
        clase = "No es un insulto (clase 6)"
    return clase
    
def modeloup(collection):
    
    #collection=collection[1:100]
    resp=[]
    for i in collection:
        temp=[]
        diccionario = nlp(i)
        X_test = loaded_tf_idf.transform([" ".join(diccionario)]).toarray()
        pred = loaded_model.predict(X_test)
        clase = tipoClase(pred)
        temp.append(i)
        temp.append(clase)
        resp.append(temp)
    
    return resp


def modelos(text,collection):
    #######################MODELO 1#############################
    diccionario = nlp(text)
    X_test = loaded_tf_idf.transform([" ".join(diccionario)]).toarray()
    pred = loaded_model.predict(X_test)
    modelo1 = loaded_model.predict_proba(X_test)

    #print('MODELO 1')
    #print(modelo1)
    ######################MODELOS DEL 2 AL 5###############################
    array = []
    for i,doc in enumerate(collection):
        array.append(re.sub('[^A-Za-z\u00C0-\u017F]+', ' ',str(collection[i].lower()))) 


    tokens = []   
    for i in range(len(array)):
        tokens.append(word_tokenize(array[i]))

    final_tokens=[]
    sw = stopwords.words("spanish")
    for i in range(len(tokens)):
        final_tokens.append([w for w in tokens[i] if not w.lower() in sw])

    dic = final_tokens

    #print(dic[0])

    D1=text
    mensaje=D1
    D1=re.sub('[^A-Za-z\u00C0-\u017F]+',' ',str(D1))
    D1=D1.lower()
    txt_tokens = word_tokenize(D1)
    vtxt = [word for word in txt_tokens if not word in stopwords.words("spanish")]


    aux=[]
    for i in range(len(dic)):
        aux.append(textdistance.jaccard(dic[i],vtxt))


    jackard_matrix=np.matrix(aux,dtype=float)


    #print('JACKARD')

    #print(jackard_matrix)



    aux1=[]
    for i in range(len(dic)):
        aux1.append(textdistance.cosine(dic[i],vtxt))


    coseno_matrix=np.matrix(aux1,dtype=float) 

    #print('COSENO')

    #print(coseno_matrix)


    aux2=[]
    for i in range(len(dic)):
        aux2.append(textdistance.tversky(dic[i],vtxt))


    tversky_matrix=np.matrix(aux2,dtype=float) 

    #print('TVERSKY')

    #print(tversky_matrix)



    aux3=[]
    for i in range(len(dic)):
        aux3.append(textdistance.tanimoto.normalized_distance(dic[i],vtxt))


    tanimoto_matrix=np.matrix(aux3,dtype=float) 

    #print('TANIMOTO')
    #print(tanimoto_matrix)
    ######################
    #probabilidades = np.concatenate((modelo1,jackard_matrix,coseno_matrix,tversky_matrix,tanimoto_matrix),axis=0)
    probabilidades = np.concatenate((modelo1,jackard_matrix,coseno_matrix,tversky_matrix,tanimoto_matrix),axis=0)

    clase = tipoClase(pred)

    colum_labels=['Clase 1','Clase 2','Clase 3','Clase 4','Clase 5','Clase 6']

    probabilidades = pd.DataFrame(probabilidades, index=index_labels, columns=colum_labels)
    return clase, probabilidades