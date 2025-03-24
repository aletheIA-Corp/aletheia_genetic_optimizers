from typing import List
from individuals import Individual
import numpy as np


class Mutation:
    def __init__(self, individual_list: List[Individual], mutate_probability: float):
        self.individual_list: List[Individual] = individual_list
        self.mutate_probability: float = mutate_probability
        self.bounds_dict = self.individual_list[0].bounds_dict

    def run_mutation(self):
        for individual in self.individual_list:
            # Realizamos los cruces de cada gen
            for parameter, bound in self.bounds_dict.items():
                match bound['bound_type']:
                    case 'predefined':
                        if np.random.rand() < self.mutate_probability:
                            individual.set_individual_value(parameter, self.mutation_bit_flip(individual, parameter))

                    case 'interval':
                        if np.random.rand() < self.mutate_probability:
                            individual.set_individual_value(parameter, self.mutation_uniform(individual, parameter))

        return self.individual_list

    def mutation_bit_flip(self, individual: Individual, parameter: str):
        """
        Metodo para mutar valores discreto


        :param individual: Indivudo que se quiere mutar alguno de sus genes
        :param parameter: Parámetro que se quiere modificar del indiviudo
        :return: Parámetro mutado.
        """

        possible_values = [z for z in self.bounds_dict[parameter]["malformation_limits"] if z != individual.get_individual_values()[parameter]]
        return float(np.random.choice(possible_values)) if self.bounds_dict[parameter]["type"] == "float" else int(np.random.choice(possible_values))

    def mutation_uniform(self, individual, parameter):
        """
        Realiza una mutación uniforme en valores enteros o reales.

        :param individual: Indivudo que se quiere mutar alguno de sus genes
        :param parameter: Parámetro que se quiere modificar del indiviudo

        :return: Parámetro mutado.
        """

        parameter_bounds: list = [z for z in self.bounds_dict[parameter]["malformation_limits"] if z != individual.get_individual_values()[parameter]]

        match self.bounds_dict[parameter]["type"]:
            case "float":
                return float(np.random.uniform(parameter_bounds[0], parameter_bounds[1]))
            case "int":
                return int(np.random.uniform(parameter_bounds[0], parameter_bounds[1]))
