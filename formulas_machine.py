import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Función para cargar datos
def load_data():
    # Permitir la carga de múltiples archivos CSV
    uploaded_files = st.sidebar.file_uploader("Elige archivos CSV :clipboard:", type="csv", key="file_uploader", accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 1:
            all_data = []
            for uploaded_file in uploaded_files:
                try:
                    # Leer cada archivo CSV
                    df = pd.read_csv(uploaded_file, sep=';', quotechar='"')
                    all_data.append(df)
                except ValueError as e:
                    st.error(f"Error al leer el archivo {uploaded_file.name}: {e}")

            # Combinar todos los DataFrames en uno solo
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                st.write("Datos combinados:")
                st.write(combined_df)
                return combined_df
        else:
            try:
                df = pd.read_csv(uploaded_files[0], sep=';', quotechar='"')
                st.write("Datos cargados:")
                st.write(df)
                return df
            except ValueError as e:
                st.error(f"Error al leer el archivo {uploaded_files[0].name}: {e}")
    
    return None

# Función para limpiar datos
def clean_data(df):
    st.write("Descripción de los datos:")
    st.write(df.describe())

    # Manejar valores faltantes
    df = df.dropna()

    # Escalar características
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

# Función para elegir modelo
def choose_model():
    model_option = st.sidebar.selectbox("Elige un modelo :smirk:", ["KNN", "Regresión Logística", "Regresión Lineal", 
                                                    "Árbol de Decisión (Clasificación)", "Árbol de Decisión (Regresión)", 
                                                    "Random Forest", "Bagging", "AdaBoosting", "Gradient Boosting"], key="model_selector")
    
    model = None
    problem_type = None
    param_grid = {}
    
    if model_option == "KNN":
        k = st.sidebar.slider("Elige el valor de K :muscle: ", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier()
        param_grid = {'n_neighbors': np.arange(1, 21)}
        problem_type = 'classification'
    elif model_option == "Regresión Logística":
        model = LogisticRegression()
        param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
        problem_type = 'classification'
    elif model_option == "Regresión Lineal":
        model = LinearRegression()
        problem_type = 'regression'
    elif model_option == "Árbol de Decisión (Clasificación)":
        max_depth = st.sidebar.slider("Max depth", 1, 20, 3)
        model = DecisionTreeClassifier()
        param_grid = {'max_depth': np.arange(1, 21)}
        problem_type = 'classification'
    elif model_option == "Árbol de Decisión (Regresión)":
        max_depth = st.sidebar.slider("Max depth", 1, 20, 3)
        model = DecisionTreeRegressor()
        param_grid = {'max_depth': np.arange(1, 21)}
        problem_type = 'regression'
    elif model_option == "Random Forest":
        n_estimators = st.sidebar.slider("Número de estimadores", 10, 100, 10)
        model = RandomForestRegressor()
        param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30]}
        problem_type = 'regression'
    elif model_option == "Bagging":
        model = BaggingRegressor()
        param_grid = {'n_estimators': [10, 50, 100]}
        problem_type = 'regression'
    elif model_option == "AdaBoosting":
        n_estimators = st.sidebar.slider("Número de estimadores", 10, 100, 10)
        model = AdaBoostRegressor()
        param_grid = {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 1]}
        problem_type = 'regression'
    elif model_option == "Gradient Boosting":
        n_estimators = st.sidebar.slider("Número de estimadores", 10, 100, 10)
        model = GradientBoostingRegressor()
        param_grid = {'n_estimators': [10, 50, 100], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 7]}
        problem_type = 'regression'
    
    return model, problem_type, param_grid

def train_and_evaluate_model(model, problem_type, param_grid, X_train, X_test, y_train, y_test):
    best_model = model
    # Dividir la pantalla en dos columnas
    col1, col2 = st.columns(2)

    # Búsqueda de Hiperparámetros
    with col1:
        if param_grid:
            st.write("### Búsqueda de Hiperparámetros")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy' if problem_type == 'classification' else 'r2')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            st.write("Mejores Parámetros:")
            st.table(grid_search.best_params_)

    # Validación Cruzada
    with col2:
        st.write("### Validación Cruzada")
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy' if problem_type == 'classification' else 'r2')  # Utiliza el mejor modelo encontrado
        st.write("Puntuaciones de Validación Cruzada:", cv_scores)
        st.write("Media de Puntuaciones de Validación Cruzada:", cv_scores.mean())

    # Train the model with the entire training set
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    if problem_type == 'classification':
        st.write("### Matriz de confusión")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        st.write("### Reporte de Clasificación")
        cr = classification_report(y_test, y_pred)
        st.text(cr)

        st.write("### Precisión del Modelo")
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy:", accuracy)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)

    else:
        st.write("### Métricas de Regresión")
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Absolute Error:", mae)
        st.write(f"Mean Squared Error:", mse)
        st.write(f"R^2 Score:", r2)

        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        st.pyplot(fig)

        st.write("### Gráfico de errores")
        errors = y_test - y_pred

        fig, ax = plt.subplots()
        sns.histplot(errors, kde=True, ax=ax)
        ax.set_title('Error Distribution')
        ax.set_xlabel('Error')
        st.pyplot(fig)

# Interfaz de Usuario Streamlit
st.title("Sistema de Recomendación de Modelos de Machine Learning :snake:")

data = load_data()

if data is not None:
    data = clean_data(data)

    # Suponiendo que la última columna es la etiqueta
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    model, problem_type, param_grid = choose_model()

    if problem_type == 'classification' and pd.api.types.is_numeric_dtype(y):
        y = pd.cut(y, bins=3, labels=["low", "medium", "high"])  # Ajustar bins y etiquetas según sea necesario

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

if st.sidebar.button("Train and Evaluate Model"):
            with st.spinner('Training and evaluating...'):
                train_and_evaluate_model(model, problem_type, param_grid, x_train, x_test, y_train, y_test)
  

