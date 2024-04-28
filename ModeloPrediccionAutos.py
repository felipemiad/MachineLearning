# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:32:45 2024
@author: johan
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
dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')
dataTesting = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTest_carListings.zip', index_col=0)

# Preprocesamiento
numeric_features = ['Year', 'Mileage']
categorical_features = ['State', 'Make', 'Model']

label_encoder = LabelEncoder()
for col in categorical_features:
    dataTraining[col + '_encoded'] = label_encoder.fit_transform(dataTraining[col])
    dataTesting[col + '_encoded'] = label_encoder.transform(dataTesting[col])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', [x + '_encoded' for x in categorical_features])
    ]
)

# Preparar datos
X_train = preprocessor.fit_transform(dataTraining[numeric_features + [x + '_encoded' for x in categorical_features]])
y_train = dataTraining['Price']
X_test = preprocessor.transform(dataTesting[numeric_features + [x + '_encoded' for x in categorical_features]])

# Entrenar modelo
model = XGBRegressor(n_estimators=100, max_depth=None, learning_rate=0.1)
model.fit(X_train, y_train)

# Guardar modelo entrenado
joblib.dump(model, 'car_price_model.pkl', compress=3)

app = Flask(__name__)
api = Api(app, version='1.0', title='Car Price Prediction API',
          description='API for predicting car prices')

ns = api.namespace('predict', description='Car Price Predictions')

parser = api.parser()
parser.add_argument('file', type=FileStorage, location='files', required=True, help='CSV file containing car details')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

#@ns.route('/')
#class CarPrice(Resource):
#    @api.doc(parser=parser)
#    def post(self):
#        args = parser.parse_args()
#        csv_file = args['file']
#        if csv_file:
#            data = pd.read_csv(csv_file)

            # Asegúrate de que las columnas necesarias están presentes
#            required_columns = ['State', 'Make', 'Model']
#            missing_columns = [col for col in required_columns if col not in data.columns]
#            if missing_columns:
#                return {'message': f'Missing columns: {missing_columns}'}, 400

            # Aplica LabelEncoder a las columnas categóricas
#            label_encoders = {col: LabelEncoder() for col in required_columns}
#            for col in required_columns:
#                data[col + '_encoded'] = label_encoders[col].fit_transform(data[col])
            
            # Prepara las características para el modelo
#            feature_columns = numeric_features + [col + '_encoded' for col in required_columns]
#            data_prepared = preprocessor.transform(data[feature_columns])

            # Predicción
#            predictions = model.predict(data_prepared)
#            return {"result": predictions.tolist()}, 200
#        else:
#            return {'message': 'File not provided'}, 400

#if __name__ == '__main__':
#    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
@ns.route('/')
class CarPrice(Resource):
    @api.doc(parser=parser)
    def post(self):
        args = parser.parse_args()
        csv_file = args['file']
        if csv_file:
            data = pd.read_csv(csv_file)

            # Asegúrate de que las columnas necesarias están presentes
            required_columns = ['State', 'Make', 'Model']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return {'message': f'Missing columns: {missing_columns}'}, 400

            # Verifica que la columna ID está presente
            if 'ID' not in data.columns:
                return {'message': 'Missing ID column'}, 400

            # Aplica LabelEncoder a las columnas categóricas
            label_encoders = {col: LabelEncoder() for col in required_columns}
            for col in required_columns:
                data[col + '_encoded'] = label_encoders[col].fit_transform(data[col])
            
            # Prepara las características para el modelo
            feature_columns = numeric_features + [col + '_encoded' for col in required_columns]
            data_prepared = preprocessor.transform(data[feature_columns])

            # Predicción
            predictions = model.predict(data_prepared)

            # Crea un DataFrame para la salida
            output_df = pd.DataFrame({
                'ID': data['ID'],
                'Prediction': predictions
            })

            # Convertir DataFrame a CSV
            result_csv = output_df.to_csv(index=False)
            return jsonify(result_csv)

        else:
            return {'message': 'File not provided'}, 400
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5001)
