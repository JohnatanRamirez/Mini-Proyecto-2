"""
    Función para procesar el conjunto de datos y crear características para el modelo de predicción.
    Se cargan los datos, se eliminan columnas innecesarias, se imputan valores faltantes,
    y se realiza la codificación de variables categóricas.
    """
import pickle
import pandas as pd

# sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

    # Cargar el dataset procesado
    dataset = pd.read_csv('../data/processed/features_for_model.csv')
    # dataset = pd.read_csv('features_for_model.csv')

    x = dataset.drop(['HighSatisfaction'], axis=1)
    y = dataset['HighSatisfaction']

    # Dividir en train y test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        shuffle=True, random_state=2025)

    # Escalar los datos
    std_scaler = StandardScaler()
    x_train_std = std_scaler.fit_transform(x_train)
    x_test_std = std_scaler.transform(x_test)

    # Guardar el scaler
    with open('../artifacts/std_scaler.pkl', 'wb') as f:
        pickle.dump(std_scaler, f)

    #with open('std_scaler.pkl', 'wb') as f:
        #pickle.dump(std_scaler, f)

    # Definir modelos e hiperparámetros
    # Modelo 1: Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=2025)
    modelo_rf.fit(x_train_std, y_train)
    y_preds_rf = modelo_rf.predict(x_test_std)
    accuracy_rf = accuracy_score(y_test, y_preds_rf)

    # Modelo 2: Regresión Logística
    modelo_rl = LogisticRegression(C=1.0, solver='liblinear', random_state=2025)
    modelo_rl.fit(x_train_std, y_train)
    y_preds_rl = modelo_rl.predict(x_test_std)
    accuracy_rl = accuracy_score(y_test, y_preds_rl)

    # Modelo 3: SVC
    modelo_svc = SVC(C=1.0, kernel='rbf', random_state=2025)
    modelo_svc.fit(x_train_std, y_train)
    y_preds_svc = modelo_svc.predict(x_test_std)
    accuracy_svc = accuracy_score(y_test, y_preds_svc)

    # Modelo 4: K-Nearest Neighbors
    modelo_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    modelo_knn.fit(x_train_std, y_train)
    y_preds_knn = modelo_knn.predict(x_test_std)
    accuracy_knn = accuracy_score(y_test, y_preds_knn)

    # Modelo 5: Árbol de Decisión
    modelo_dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
                                       random_state=2025)
    y_preds_dt = modelo_dt.predict(x_test_std)
    accuracy_dt = accuracy_score(y_test, y_preds_dt)

    # Comparar resultados
    resultados = {
        'RandomForest': accuracy_rf,
        'LogisticRegression': accuracy_rl,
        'SVC': accuracy_svc,
        'KNeighbors': accuracy_knn,
        'DecisionTree': accuracy_dt
    }

    print("Resultados de precisión por modelo:")
    for modelo, accuracy in resultados.items():
        print(f"{modelo}: {accuracy:.4f}")

    with open('../models/random_forest_v1.pkl', 'wb') as f:
        pickle.dump(modelo_rf,f)

    #Aqui se guarda el modelo que haya dado los mejores resultados, en mi caso Random Forest
    #with open('random_forest_v1.pkl', 'wb') as f:
        #pickle.dump(modelo_rf,f)
