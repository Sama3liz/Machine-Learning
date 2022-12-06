from flask import Flask,render_template,request, url_for, redirect, send_from_directory
from pymongo import MongoClient
import pandas as pd
import numpy as np
import static.py.modelos as m
import static.py.functions as f
import os
from werkzeug.utils import secure_filename

# Config
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/files"

# DB Connection
client = MongoClient("mongodb+srv://prueba:6k9Zll2W0hoLMenG@cluster0.dutodri.mongodb.net/?retryWrites=true&w=majority")
db = client.dataset
collection = db.clases

# Routes
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        clase1=collection.distinct('clase1')
        clase2=collection.distinct('clase2')
        clase3=collection.distinct('clase3')
        clase4=collection.distinct('clase4')
        clase5=collection.distinct('clase5')
        clase6=collection.distinct('clase6')
        clase1=" ".join(clase1[0:505])
        clase2=" ".join(clase2[0:505])
        clase3=" ".join(clase3[0:499])
        clase4=" ".join(clase4[0:505])
        clase5=" ".join(clase5[0:505])
        clase6=" ".join(clase6[0:5299])
        col = [clase1,clase2,clase3,clase4,clase5,clase6]
        respuesta = m.modelos(text,col)
        df = respuesta[1]
        return render_template('home.html', value = respuesta[0], tables=[df.to_html(classes='data', header="true")])
    else:
        return render_template('home.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        file = request.files['uploadFile']
        filename = secure_filename(file.filename)
        if file and f.allowed_file(filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file=os.path.join(app.config["UPLOAD_FOLDER"],filename)
            excel_data = pd.read_excel(file)
            arr=excel_data.to_numpy()
            data=m.modeloup(arr)
            analisis = pd.DataFrame(data, columns=['VALOR DE INGRESO','CLASIFICACION'])
            analisis.to_excel(os.path.join(app.config["UPLOAD_FOLDER"],'analisis.xlsx')) 
            #BORRAR ARCHIVO
            os.remove(file)
            resp_analisis = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
            
            return send_from_directory(directory=resp_analisis, path='analisis.xlsx')
        else:
            print("Tipo de archivo no permitido!")
            return redirect(url_for('upload'))
    else:
        return render_template('upload.html')

@app.route("/about", methods=['GET','POST'])
def about():
    if request.method == 'POST':
        return render_template('about.html')
    else:
        return render_template('about.html')
    
if __name__ == '__main__':
    app.run(debug=True)
