# -*- coding: utf-8 -*-
"""
Created on Sun May 26 09:32:29 2024

@author: johan
"""

import warnings
warnings.filterwarnings('ignore')

# Importación librerías
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import joblib
import ast

# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTraining.zip', encoding='UTF-8', index_col=0)
dataTesting = pd.read_csv('https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/dataTesting.zip', encoding='UTF-8', index_col=0)

# Definición de variables predictoras (X)
vect = CountVectorizer(max_features=1000)
X_dtm = vect.fit_transform(dataTraining['plot'])

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

# Definición de variable de interés (y)
dataTraining['genres'] = dataTraining['genres'].map(lambda x: safe_literal_eval(x) if isinstance(x, str) else x)

le = MultiLabelBinarizer()
y_genres = le.fit_transform(dataTraining['genres'])

# Separación de variables predictoras (X) y variable de interés (y) en set de entrenamiento y test usando la función train_test_split
X_train, X_test, y_train_genres, y_test_genres = train_test_split(X_dtm, y_genres, test_size=0.33, random_state=42)

# Definición y entrenamiento
clf = OneVsRestClassifier(RandomForestClassifier(n_jobs=-1, n_estimators=100, max_depth=10, random_state=42))
clf.fit(X_train, y_train_genres)

# Guardar el modelo entrenado, vectorizador y label encoder
joblib.dump(clf, 'genre_classification_model.pkl')
joblib.dump(vect, 'vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Evaluación del modelo
y_pred_genres = clf.predict_proba(X_test)
print('ROC AUC Score:', roc_auc_score(y_test_genres, y_pred_genres, average='macro'))

