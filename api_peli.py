# -*- coding: utf-8 -*-
"""
Created on Sun May 26 09:33:05 2024

@author: johan
"""

from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import pandas as pd
import joblib
from werkzeug.datastructures import FileStorage
import numpy as np

# Cargar el modelo, vectorizador y label encoder
clf = joblib.load('genre_classification_model.pkl')
vect = joblib.load('vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

app = Flask(__name__)
api = Api(app, version='1.0', title='API Genero de Peliculas',
          description='API para clasificar géneros de películas')

ns = api.namespace('predict', description='Predicción de Género de Peliculas')

parser = api.parser()
parser.add_argument('file', type=FileStorage, location='files', required=True, help='archivo CSV que contiene caracteristicas de cada pelicula')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class GenrePrediction(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        archivo_csv = args['file']
        if archivo_csv:
            # Leer el archivo CSV y asignar nombres a las columnas
            data = pd.read_csv(archivo_csv, names=['ID', 'year', 'title', 'plot'], header=0)

            # Verificar las columnas del DataFrame
            print("Columnas del DataFrame:", data.columns.tolist())

            # Asegúrate de que la columna 'plot' está presente
            if 'plot' not in data.columns:
                return {'message': 'Missing plot column'}, 400

            # Preprocesamiento del texto
            X_dtm = vect.transform(data['plot'])

            # Predicción
            predictions = clf.predict_proba(X_dtm)

            # Crear un DataFrame para la salida
            cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
                    'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
                    'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

            output_df = pd.DataFrame(predictions, columns=cols)
            output_df.insert(0, 'ID', data['ID'])

            # Convertir DataFrame a CSV
            result_csv = output_df.to_csv(index=False)
            return result_csv

        else:
            return {'message': 'File not provided'}, 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)

