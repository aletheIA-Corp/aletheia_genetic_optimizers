from individuals import Individual
from bounds import BoundCreator
from genetic_optimizer import GenethicOptimizer

# -- Definimos la función objetivo
def objective_function(individual: Individual) -> float:
    import numpy as np
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # -- Cargamos el dataset de diabetes
    data = load_diabetes()
    individual_dict: dict = individual.get_individual_values()
    X, y = data.data, data.target

    # -- Convertimos la variable objetivo en un problema de clasificación binaria (diabetes alta o baja)
    y = (y > np.median(y)).astype(int)  # 1 si es mayor a la mediana, 0 si es menor

    # -- Dividimos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -- Normalizamos los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Función objetivo para entrenar el modelo y calcular la precisión
    def train_and_evaluate_model(individual_dict, X_train_scaled, X_test_scaled, y_train, y_test):
        model = RandomForestClassifier(n_estimators=int(individual_dict["n_estimators"]),
                                       max_depth=individual_dict["max_depth"],
                                       random_state=42)

        # -- Entrenamos el modelo, predecimos y calculamos el accuracy
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return accuracy_score(y_test, y_pred)

    # -- Entrenamos y evaluamos el modelo
    accuracy = train_and_evaluate_model(individual_dict, X_train_scaled, X_test_scaled, y_train, y_test)

    return accuracy


def example_1_bounds_no_predefinidos():
    # -- Creamos el diccionario de bounds
    bounds = BoundCreator()
    bounds.add_interval_bound("n_estimators", 100, 1000, 50, 1500, "int")
    bounds.add_predefined_bound("max_depth", (1, 2, 3, 4, 5, 6, 7, 8, 9), "int")

    print(GenethicOptimizer(bounds.get_bound(),
                            5,
                            9,
                            objective_function,
                            "maximize",
                            "ea_simple",
                            3,
                            0.2,
                            0.25,
                            0.0,
                            0.5,
                            ))


example_1_bounds_no_predefinidos()
