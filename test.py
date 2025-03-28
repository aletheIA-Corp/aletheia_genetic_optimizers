import datetime

from aletheia_genetic_optimizers import Individual
from aletheia_genetic_optimizers import BoundCreator
from aletheia_genetic_optimizers import GenethicOptimizer
import numpy as np


# -- Definimos la función objetivo
def example_1_bounds_no_predefinidos():
    import numpy as np
    import datetime
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    def objective_function(individual):
        # Load diabetes dataset
        data = load_diabetes()
        individual_dict = individual.get_individual_values()
        X, y = data.data, data.target

        # Convert target variable to binary classification
        y = (y > np.median(y)).astype(int)  # 1 if above median, 0 if below

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Measure execution time
        start_time = datetime.datetime.now()

        # Normalize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def train_and_evaluate_model(individual_dict, X_train_scaled, X_test_scaled, y_train, y_test):
            model = RandomForestClassifier(
                n_estimators=int(individual_dict["n_estimators"]),
                max_depth=individual_dict["max_depth"],
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            return accuracy_score(y_test, y_pred)

        # Calculate accuracy
        accuracy = train_and_evaluate_model(individual_dict, X_train_scaled, X_test_scaled, y_train, y_test)

        # Calculate execution time
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

        # Penalization parameters
        reference_time = 0.5  # Reference time for penalization
        max_penalty_ratio = 0.01  # Maximum penalty (1% of accuracy)

        # Calculate continuous time-based penalty
        # The penalty increases smoothly as time increases
        time_penalty = max_penalty_ratio * accuracy * (elapsed_time / reference_time)
        time_penalty = min(time_penalty, max_penalty_ratio * accuracy)

        penalized_accuracy = accuracy - time_penalty

        # print(f"[Original]: {accuracy}   ---   [Penalized]: {penalized_accuracy}")

        return penalized_accuracy

    # -- Creamos el diccionario de bounds
    bounds = BoundCreator()
    bounds.add_interval_bound("n_estimators", 50, 1000, 10, 1500, "int")
    bounds.add_predefined_bound("max_depth", (1, 2, 3, 4, 5, 6, 7, 8, 9), "int")

    return GenethicOptimizer(bounds.get_bound(),
                             30,
                             30,
                             objective_function,
                             "bound_restricted",
                             "maximize",
                             "ea_simple",
                             3,
                             0.3,
                             0.3,
                             )

def example_2_tsp():
    # Coordenadas de ciudades (ejemplo con 5 ciudades)

    def objective_function(individual):
        original_cities = {
            0: (4.178515558765522, 3.8658505110962347),
            1: (9.404615166221248, 8.398020682045034),
            2: (0.3782334121284714, 8.295288013706802),
            3: (9.669161753695562, 5.593501025856912),
            4: (9.870966532678576, 4.756484445482374),
            5: (3.5045826424785007, 1.1043994011149494),
            6: (5.548867108083866, 5.842473649079045),
            7: (1.11377627026643, 1.304647970128091),
            8: (5.133591646349645, 3.8238217557909038),
            9: (7.074655346940579, 3.6554091142752734),
            10: (9.640123872995837, 1.3285594561699254),
            11: (0.021205320973052277, 7.018385604153457),
            12: (2.048903069073358, 2.562383464533476),
            13: (2.289964825687684, 4.325937821712228),
            14: (6.315627335092245, 3.7506598107821656),
            15: (1.0589427543395036, 6.2520630725232),
            16: (9.218474645470067, 4.106769373018785),
            17: (4.62163288328154, 9.583091224200263),
            18: (7.477615269848112, 7.597659062497909),
            19: (0.25092704950321565, 6.699275814039302),
        }

        # Obtener valores únicos en route y ordenarlos
        route = list(individual.get_individual_values().values())
        unique_cities = sorted(set(route))  # Se asegura de tener un orden fijo

        # Crear nuevo índice de ciudades dinámico
        cities = {idx: original_cities[city] for idx, city in enumerate(unique_cities)}

        num_cities = len(cities)
        distance_matrix = np.zeros((num_cities, num_cities))

        # Calcular la matriz de distancias con los nuevos índices
        for i in range(num_cities):
            for j in range(num_cities):
                if i != j:
                    x1, y1 = cities[i]
                    x2, y2 = cities[j]
                    distance_matrix[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        # Convertir route a los nuevos índices
        route_mapped = [unique_cities.index(city) for city in route]

        # Calcular distancia total
        total_distance = sum(
            distance_matrix[route_mapped[i]][route_mapped[(i + 1) % len(route_mapped)]]
            for i in range(len(route_mapped))
        )

        return total_distance

    # -- Creamos el diccionario de bounds
    range_list: tuple = tuple([z for z in range(0, 20)])
    bounds = BoundCreator()
    bounds.add_predefined_bound("city_zero", range_list, "int")
    bounds.add_predefined_bound("city_one", range_list, "int")
    bounds.add_predefined_bound("city_two", range_list, "int")
    bounds.add_predefined_bound("city_three", range_list, "int")
    bounds.add_predefined_bound("city_four", range_list, "int")
    bounds.add_predefined_bound("city_five", range_list, "int")
    bounds.add_predefined_bound("city_six", range_list, "int")
    bounds.add_predefined_bound("city_seven", range_list, "int")
    bounds.add_predefined_bound("city_eight", range_list, "int")
    bounds.add_predefined_bound("city_nine", range_list, "int")
    bounds.add_predefined_bound("city_ten", range_list, "int")
    bounds.add_predefined_bound("city_eleven", range_list, "int")
    bounds.add_predefined_bound("city_twelve", range_list, "int")
    bounds.add_predefined_bound("city_trece", range_list, "int")
    bounds.add_predefined_bound("city_14", range_list, "int")
    bounds.add_predefined_bound("city_15", range_list, "int")
    bounds.add_predefined_bound("city_16", range_list, "int")
    bounds.add_predefined_bound("city_17", range_list, "int")
    bounds.add_predefined_bound("city_18", range_list, "int")
    bounds.add_predefined_bound("city_19", range_list, "int")

    return GenethicOptimizer(bounds.get_bound(),
                             500,
                             100,
                             objective_function,
                             "full_restricted",
                             "minimize",
                             "ea_simple",
                             3,
                             0.25,
                             0.1,
                             'medium',
                             True
                             )


genetic_optimizer_object: GenethicOptimizer = example_1_bounds_no_predefinidos()
genetic_optimizer_object.plot_generation_stats()
# genetic_optimizer_object.plot_evolution_animated()
genetic_optimizer_object.plot_evolution()
