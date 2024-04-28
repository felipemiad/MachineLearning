# -*- coding: utf-8 -*-
"""
Creado el Sun Apr 28 13:32:45 2024
@autor: johan
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
from xgboost import XGBRegressor

# Carga de datos de entrenamiento y prueba
datosEntrenamiento = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
datosPrueba = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)

# Preprocesamiento
caracteristicas_numericas = ['Year', 'Mileage']
caracteristicas_categoricas = ['State', 'Make', 'Model']

codificador_etiquetas = LabelEncoder()
for col in caracteristicas_categoricas:
    datosEntrenamiento[col + '_codificado'] = codificador_etiquetas.fit_transform(datosEntrenamiento[col])
    datosPrueba[col + '_codificado'] = codificador_etiquetas.transform(datosPrueba[col])

preprocesador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), caracteristicas_numericas),
        ('cat', 'passthrough', [x + '_codificado' for x in caracteristicas_categoricas])
    ]
)

# Preparar datos
X_entrenamiento = preprocesador.fit_transform(datosEntrenamiento[caracteristicas_numericas + [x + '_codificado' for x in caracteristicas_categoricas]])
y_entrenamiento = datosEntrenamiento['Price']
X_prueba = preprocesador.transform(datosPrueba[caracteristicas_numericas + [x + '_codificado' for x in caracteristicas_categoricas]])

# Entrenar modelo
modelo = XGBRegressor(n_estimators=100, max_depth=None, learning_rate=0.1)
modelo.fit(X_entrenamiento, y_entrenamiento)

# Guardar modelo entrenado
joblib.dump(modelo, 'modelo_precio_vehiculo.pkl', compress=3)

app = Flask(__name__)
api = Api(app, version='1.0', titulo='API MIAD Predicción de precio de Vehículos ML',
          descripcion='API para la predicción del precio de un automóvil según sus características')

ns = api.namespace('predict', descripcion='Predicciones de precios de vehículos')

analizador = api.parser()
analizador.add_argument('archivo', type=FileStorage, location='files', required=True, help='Archivo CSV con detalles del vehículo')

campos_recurso = api.model('Recurso', {
    'resultado': fields.String,
})

@ns.route('/')
class PrecioVehiculo(Resource):
    @api.doc(parser=analizador)
    def post(self):
        args = analizador.parse_args()
        archivo_csv = args['archivo']
        if archivo_csv:
            datos = pd.read_csv(archivo_csv)

            # Asegúrate de que las columnas necesarias están presentes
            columnas_necesarias = ['State', 'Make', 'Model']
            columnas_faltantes = [col for col in columnas_necesarias if col not in datos.columns]
            if columnas_faltantes:
                return {'mensaje': f'Columnas faltantes: {columnas_faltantes}'}, 400

            # Aplica LabelEncoder a las columnas categóricas
            codificadores_etiquetas = {col: LabelEncoder() for col in columnas_necesarias}
            for col in columnas_necesarias:
                datos[col + '_codificado'] = codificadores_etiquetas[col].fit_transform(datos[col])
            
            # Prepara las características para el modelo
            columnas_caracteristicas = caracteristicas_numericas + [col + '_codificado' for col in columnas_necesarias]
            datos_preparados = preprocesador.transform(datos[columnas_caracteristicas])

            # Predicción
            predicciones = modelo.predict(datos_preparados)
            return {"resultado": predicciones.tolist()}, 200
        else:
            return {'mensaje': 'Archivo no proporcionado'}, 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
