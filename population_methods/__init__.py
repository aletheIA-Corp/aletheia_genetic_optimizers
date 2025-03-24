import pandas as pd

from individuals import Individual
from typing import List, Dict, Tuple, Union
from info_tools import InfoTools
import scipy.stats as stats
import numpy as np


class Population:
    def __init__(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], num_individuals: int):
        self.IT: InfoTools = InfoTools
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict
        self.num_individuals: int = num_individuals
        self.populuation_dict: Dict[str, List[Individual]] = {0: []}
        self.generations_fitness_statistics_df: pd.DataFrame | None = None

    def create_population(self) -> None:
        """
        Método para crear la población inicial
        :return:
        """
        while len(self.populuation_dict[0]) < self.num_individuals:
            individual: Individual = Individual(self.bounds_dict, None, 0)
            if not individual.malformation:
                self.populuation_dict[0].append(individual)

    def add_generation_population(self, children_list: List[Individual], generation: int) -> None:
        """
        Método para agregar una generación de individuos al diccionario de población
        :param children_list:
        :param generation:
        :return:
        """
        self.populuation_dict[generation] = children_list

    def get_generation_fitness_statistics(self, generation: int):
        """
            Calcula estadísticas descriptivas de una lista de valores numéricos.

            :param generation: Generación de la que vamos a obtener la info
            :return: Diccionario con media, mediana, desviación estándar, cuartiles, rango, moda y más.
            """
        values = np.array([z.individual_fitness for z in self.populuation_dict[generation] if not z.malformation])

        data = {
            "generation": [generation],
            "count": [len(values)],
            "mean": [np.mean(values)],
            "median": [np.median(values)],
            "std_dev": [np.std(values, ddof=1)],
            "variance": [np.var(values, ddof=1)],
            "min": [np.min(values)],
            "max": [np.max(values)],
            "range": [np.ptp(values)],
            "q1": [np.percentile(values, 25)],
            "q2": [np.percentile(values, 50)],  # Mediana
            "q3": [np.percentile(values, 75)],
            "iqr": [stats.iqr(values)],
            "mode": [stats.mode(values, keepdims=True)[0][0]]
        }

        if self.generations_fitness_statistics_df is None:
            self.generations_fitness_statistics_df = pd.DataFrame(data)
        else:
            self.generations_fitness_statistics_df = pd.concat([self.generations_fitness_statistics_df, pd.DataFrame(data)], axis=0)

        return self.generations_fitness_statistics_df
