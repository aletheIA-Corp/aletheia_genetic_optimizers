from typing import List, Literal
from aletheia_genetic_optimizers.individuals import Individual
import numpy as np
import random


class Mutation:
    def __init__(self, individual_list: List[Individual],
                 mutate_probability: float,
                 mutate_gen_probability: float,
                 mutation_policy: Literal['soft', 'normal', 'hard'],
                 problem_restrictions: Literal['bound_restricted', 'full_restricted']):

        self.individual_list: List[Individual] = individual_list
        self.mutate_probability: float = mutate_probability
        self.mutate_gen_probability: float = mutate_gen_probability
        self.mutation_policy: Literal['soft', 'normal', 'hard'] = mutation_policy
        # -- Almaceno en una propiedad las restricciones a aplicar
        self.problem_restrictions: Literal['bound_restricted', 'full_restricted'] = problem_restrictions
        self.bounds_dict = self.individual_list[0].bounds_dict

    def run_mutation(self):
        match self.problem_restrictions:
            case "bound_restricted":
                return self.bound_restricted_mutation()
            case "full_restricted":
                return self.full_restricted_mutation()

    # <editor-fold desc="Mutaciones en funcion del self.problem_restrictions    --------------------------------------------------------------------------------------------------">

    def bound_restricted_mutation(self):
        for individual in self.individual_list:
            if np.random.rand() >= self.mutate_probability:
                continue
            # Realizamos los cruces de cada gen
            for parameter, bound in self.bounds_dict.items():
                match bound['bound_type']:
                    case 'predefined':
                        if np.random.rand() < self.mutate_gen_probability:
                            individual.set_individual_value(parameter, self.mutation_bit_flip(individual, parameter))

                    case 'interval':
                        if np.random.rand() < self.mutate_gen_probability:
                            individual.set_individual_value(parameter, self.mutation_uniform(individual, parameter))

        return self.individual_list

    def full_restricted_mutation(self):
        """Realiza una mutación por intercambio en la lista dada."""
        # TODO: Implementar potencia de la mutacion con self.mutation_policy
        for individual in self.individual_list:
            if np.random.rand() >= self.mutate_probability:
                continue

            individual_values = [z for z in individual.get_individual_values().values()]

            if len(individual_values) < 2:
                return individual_values  # No se puede mutar si hay menos de 2 elementos
            i, j = random.sample(range(len(individual_values)), 2)  # Escoge dos índices distintos
            individual_values[i], individual_values[j] = individual_values[j], individual_values[i]  # Intercambia los valores

            individual.set_individual_values(individual_values)

        return self.individual_list

    # </editor-fold>

    # <editor-fold desc="Metodos de mutacion de genes    -------------------------------------------------------------------------------------------------------------------------">

    def mutation_bit_flip(self, individual: Individual, parameter: str):
        """
        Metodo para mutar valores discreto


        :param individual: Indivudo que se quiere mutar alguno de sus genes
        :param parameter: Parámetro que se quiere modificar del indiviudo
        :return: Parámetro mutado.
        """

        # TODO: Implementar potencia de la mutacion con self.mutation_policy

        possible_values = [z for z in self.bounds_dict[parameter]["malformation_limits"] if z != individual.get_individual_values()[parameter]]
        return float(np.random.choice(possible_values)) if self.bounds_dict[parameter]["type"] == "float" else int(np.random.choice(possible_values))

    def mutation_uniform(self, individual, parameter):
        """
        Realiza una mutación uniforme en valores enteros o reales.

        :param individual: Indivudo que se quiere mutar alguno de sus genes
        :param parameter: Parámetro que se quiere modificar del indiviudo

        :return: Parámetro mutado.
        """
        # TODO: Implementar potencia de la mutacion con self.mutation_policy
        parameter_bounds: list = [z for z in self.bounds_dict[parameter]["malformation_limits"] if z != individual.get_individual_values()[parameter]]

        match self.bounds_dict[parameter]["type"]:
            case "float":
                return float(np.random.uniform(parameter_bounds[0], parameter_bounds[1]))
            case "int":
                return int(np.random.uniform(parameter_bounds[0], parameter_bounds[1]))

    # </editor-fold>