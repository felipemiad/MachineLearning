# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:15:02 2024

@author: johan
"""

import warnings
warnings.filterwarnings('ignore')

# Importación librerías
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

import os
os.chdir('..')
# Carga de datos de archivos .csv
data = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/phishing.csv')
data.head()

# Creación de columnas binarias que indican si la URL contiene la palabra clave (keywords)
keywords = ['https', 'login', '.php', '.html', '@', 'sign']
for keyword in keywords:
    data['keyword_' + keyword] = data.url.str.contains(keyword).astype(int)

# Definición de la variable largo de la URL
data['lenght'] = data.url.str.len() - 2

# Definición de la variable largo del dominio de la URL
domain = data.url.str.split('/', expand=True).iloc[:, 2]
data['lenght_domain'] = domain.str.len()

# Definición de la variable binaria que indica si es IP
data['isIP'] = (domain.str.replace('.', '') * 1).str.isnumeric().astype(int)

# Definicón de la variable cuenta de 'com' en la URL
data['count_com'] = data.url.str.count('com')

data.head()

# Separación de variables predictoras (X) y variable de interes (y)
X = data.drop(['url', 'phishing'], axis=1)
y = data.phishing

# Definición de modelo de clasificación Random Forest
clf = RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=3)
cross_val_score(clf, X, y, cv=10)

# Entrenamiento del modelo de clasificación Random Forest
clf.fit(X, y)

# Exportar modelo a archivo binario .pkl
joblib.dump(clf, 'model_deployment/phishing_clf.pkl', compress=3)
# Importar modelo y predicción
from model_deployment.m09_model_deployment import predict_proba

# Predicción de probabilidad de que un link sea phishing
predict_proba('http://www.vipturismolondres.com/com.br/?atendimento=Cliente&/LgSgkszm64/B8aNzHa8Aj.php')

# Importación librerías
from flask import Flask
from flask_restx import Api, Resource, fields
# Definición aplicación Flask
app = Flask(__name__)

# Definición API Flask
api = Api(
    app, 
    version='1.0', 
    title='Phishing Prediction API',
    description='Phishing Prediction API')

ns = api.namespace('predict', 
     description='Phishing Classifier')

# Definición argumentos o parámetros de la API
parser = api.parser()
parser.add_argument(
    'URL', 
    type=str, 
    required=True, 
    help='URL to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})
# Definición de la clase para disponibilización
@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['URL'])
        }, 200
# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)