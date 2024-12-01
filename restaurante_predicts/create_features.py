"""
Este modulo contiene las funciones para aplicar la ingenieria de caracteristicas
"""
import pickle
import pandas as pd

def create_model_features():
    """_summary_: esta funcion crea los features del modelo
    """
    ### 1. Cargamos Datos
    dataset = pd.read_csv("../data/raw/train.csv")

    ### 3. Eliminamos variables no útiles
    dataset.drop(['CustomerID'], axis=1, inplace=True)

    ### 4. Ingeniería de Características
    # Codificación de las variables usando Label Encoder
    from sklearn.preprocessing import LabelEncoder

    columnas_label_encoding = ['Gender', 'VisitFrequency', 'PreferredCuisine',
                               'TimeOfVisit', 'DiningOccasion', 'MealType']

    label_encoder = LabelEncoder()
    for col in columnas_label_encoding:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    ### 5. Guardar el dataset procesado
    dataset.to_csv('../data/processed/features_for_model.csv', index=False)

    # Columnas que fueron transformadas con LabelEncoder
    columnas_label_encoding = ['Gender', 'VisitFrequency', 'PreferredCuisine',
                               'TimeOfVisit', 'DiningOccasion', 'MealType']

    # Diccionario que almacenará las clases de cada columna
    feature_eng_configs = {'label_encoders': {}}

    for col in columnas_label_encoding:
        # Recuperamos las clases del LabelEncoder aplicado
        le = LabelEncoder()
        le.fit(dataset[col])  # Reajustamos el encoder con los valores ya codificados
        feature_eng_configs['label_encoders'][col] = le.classes_.tolist()

    # Guardamos las configuraciones en un archivo pickle
    with open('../artifacts/feature_eng_configs.pkl', 'wb') as f:
        pickle.dump(feature_eng_configs, f)
