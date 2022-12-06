import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ALLOWED_EXTENSIONS = set(["xlsx"])

def allowed_file(file):
    file = file.split('.')
    if file[1] in ALLOWED_EXTENSIONS:
        return True
    else: 
        return False

def nlp(text):
    text = " ".join(text)
    text = re.sub('[^A-Za-z\u00C0-\u017F]+',' ',str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    diccionario = [word for word in tokens if not word in stopwords.words("spanish")]
    return diccionario

def readData(file):
    df = pd.read_excel(file)
    name = df.columns
    array = np.array(df)
    dic = []
    for word in array:
        dic.append(nlp(word))
    return dic
    
def justData(file):
    df = pd.read_excel(file)
    array = np.array(df)
    dic = []
    for word in array:
        dic.append(" ".join(word))
    return dic
