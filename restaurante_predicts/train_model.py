"""
_summary_: este código genera los modelos de clasificación para el problema de titanic
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

def train_model():
    """
    _summary_: esta función entrena los modelos de clasificación y almacena el mejor.
    """
    # Cargar el dataset procesado
    dataset = pd.read_csv('../data/processed/features_for_model.csv')

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

    # Definir modelos e hiperparámetros
    # Modelo 1: Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=150,
                                       max_depth=20, random_state=2025)
    modelo_rf.fit(x_train_std, y_train)
    y_preds_rf = modelo_rf.predict(x_test_std)
    accuracy_rf = accuracy_score(y_test, y_preds_rf)

    # Modelo 2: Regresión Logística
    modelo_rl = LogisticRegression(C=0.5, solver='saga', random_state=2025)
    modelo_rl.fit(x_train_std, y_train)
    y_preds_rl = modelo_rl.predict(x_test_std)
    accuracy_rl = accuracy_score(y_test, y_preds_rl)

    # Modelo 3: SVC
    modelo_svc = SVC(C=10.0, kernel='poly', random_state=2025)
    modelo_svc.fit(x_train_std, y_train)
    y_preds_svc = modelo_svc.predict(x_test_std)
    accuracy_svc = accuracy_score(y_test, y_preds_svc)

    # Modelo 4: K-Nearest Neighbors
    modelo_knn = KNeighborsClassifier(n_neighbors=7, weights='uniform')
    modelo_knn.fit(x_train_std, y_train)
    y_preds_knn = modelo_knn.predict(x_test_std)
    accuracy_knn = accuracy_score(y_test, y_preds_knn)

    # Modelo 5: Árbol de Decisión
    modelo_dt = DecisionTreeClassifier(max_depth=15,
                                       min_samples_split=5, random_state=2025)
    y_preds_dt = modelo_dt.predict(x_test_std)
    accuracy_dt = accuracy_score(y_test, y_preds_dt)

    resultados = {
    'RandomForest': accuracy_rf,
    'LogisticRegression': accuracy_rl,
    'SVC': accuracy_svc,
    'KNeighbors': accuracy_knn,
    'DecisionTree': accuracy_dt
    }

    with open('../models/random_forest_v1.pkl', 'wb') as f:
        pickle.dump(modelo_rf,f)
