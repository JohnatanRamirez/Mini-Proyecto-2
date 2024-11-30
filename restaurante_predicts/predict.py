"""
    Función para procesar el conjunto de datos y crear características para el modelo de predicción.
    Se cargan los datos, se eliminan columnas innecesarias, se imputan valores faltantes,
    y se realiza la codificación de variables categóricas.
    """

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

def create_model_feature():

    """
    Procesa el conjunto de datos para crear características
    que se usarán en el modelo de predicción.
    Operaciones realizadas:
    - Carga el conjunto de datos.
    - Elimina columnas irrelevantes ('PassengerId', 'Name', 'Ticket', 'Cabin').
    - Imputa valores faltantes en las columnas 'Age' y 'Embarked'.
    - Realiza la codificación de las variables 'Sex' y 'Embarked'.
    - Guarda el dataset procesado en un archivo CSV.
    - Almacena las configuraciones de imputación y codificación en un archivo pickle.
    """

    # Cargar dataset de prueba
    data_test = pd.read_csv('../data/raw/test.csv')

    #data_test = pd.read_csv('test.csv')

    # Cargar configuraciones de ingeniería de características
    #with open('../artifacts/feature_eng_configs.pkl', 'rb') as f:
        #feature_eng_configs = pickle.load(f)

    with open('feature_eng_configs.pkl', 'rb') as f:
        feature_eng_configs = pickle.load(f)

    # Aplicar LabelEncoder a las columnas categóricas
    columnas_label_encoding = feature_eng_configs['label_encoders'].keys()

    for col in columnas_label_encoding:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = feature_eng_configs['label_encoders'][col]  # Cargar las clases del pickle
        # Validar que todos los valores en la columna están dentro de las clases conocidas
        data_test[col] = data_test[col].apply(lambda x: x if x in label_encoder.classes_ else None)
        # Transformar solo los valores válidos
        data_test[col] = data_test[col].map(lambda x: label_encoder.transform([x])[0] if x is not None else -1)



    # Otra opción


    data_test.drop(['CustomerID'], axis=1, inplace=True)

    columnas_label_encoding = ['Gender', 'VisitFrequency', 'PreferredCuisine', 'TimeOfVisit',
                               'DiningOccasion', 'MealType']
    label_encoder = LabelEncoder()
    for col in columnas_label_encoding:
        data_test[col] = label_encoder.fit_transform(data_test[col])

    data_test = data_test.drop(columns=['HighSatisfaction'])

    # Cargar el scaler estándar si fue usado previamente
    with open('../artifacts/std_scaler.pkl', 'rb') as f:
        std_scaler = pickle.load(f)

    #with open('std_scaler.pkl', 'rb') as f:
        #std_scaler = pickle.load(f)

    # Estandarizar las variables del dataset de prueba
    X_data_test_std = std_scaler.transform(data_test)

    # Cargar el modelo guardado
    with open('../models/random_forest_v1.pkl', 'rb') as f:
        modelo = pickle.load(f)

    #with open('random_forest_v1.pkl', 'rb') as f:
        #modelo = pickle.load(f)

    # Realizar predicciones
    model_predicts = modelo.predict(X_data_test_std)

    # Mostrar las predicciones
    print("Predicciones del modelo:")
    print(model_predicts)
