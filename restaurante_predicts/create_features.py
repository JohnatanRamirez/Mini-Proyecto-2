"""
    Función para procesar el conjunto de datos y crear características para el modelo de predicción.
    Se cargan los datos, se eliminan columnas innecesarias, se imputan valores faltantes,
    y se realiza la codificación de variables categóricas.
    """

import pandas as pd

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

    # ### 1. Cargamos Datos

    dataset = pd.read_csv("../data/raw/train.csv")
    #dataset = pd.read_csv("train.csv")

    # ### 2. Exploración de datos

    # ### 3. Eliminamos variables no útiles

    dataset.drop(['CustomerID'], axis=1, inplace=True)

    # ### 4. Ingeniería de Características

    dataset.isnull().mean()

    # Codificación de las variables usando Label Encoder
    from sklearn.preprocessing import LabelEncoder
    columnas_label_encoding = ['Gender', 'VisitFrequency', 'PreferredCuisine', 'TimeOfVisit',
                                'DiningOccasion', 'MealType']

    label_encoder = LabelEncoder()
    for col in columnas_label_encoding:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    # ### 5. Guardar el dataset procesado

    dataset.to_csv('../data/processed/features_for_model.csv', index=False)
    ## dataset.to_csv('features_for_model.csv', index=False)


    import pickle

    # Columnas que fueron transformadas con LabelEncoder
    columnas_label_encoding = ['Gender', 'VisitFrequency', 'PreferredCuisine', 'TimeOfVisit',
                                'DiningOccasion', 'MealType']
    # Diccionario que almacenará las clases de cada columna
    feature_eng_configs = {'label_encoders': {}}

    for col in columnas_label_encoding:
        # Recuperamos las clases del LabelEncoder aplicado
        label_encoder= LabelEncoder()
        label_encoder.fit(dataset[col])  # Reajustamos el encoder con los valores ya codificados
        feature_eng_configs['label_encoders'][col] = label_encoder.classes_.tolist()

    # Guardamos las configuraciones en un archivo pickle
    with open('../artifacts/feature_eng_configs.pkl', 'wb') as pickle_file:
        pickle.dump(feature_eng_configs, pickle_file)

    # with open('feature_eng_configs.pkl', 'wb') as f:
        # pickle.dump(feature_eng_configs, f)
