from typing import List, Literal
from aletheia_genetic_optimizers.individuals import Individual
import numpy as np
import random


class Mutation:
    def __init__(self, individual_list: List[Individual],
                 mutate_probability: float,
                 mutate_gen_probability: float,
                 mutation_policy: Literal['soft', 'normal', 'hard'],
                 problem_restrictions: Literal['bound_restricted', 'full_restricted'],
                 num_generations: int):

        self.individual_list: List[Individual] = individual_list
        self.mutate_probability: float = mutate_probability
        self.mutate_gen_probability: float = mutate_gen_probability
        self.mutation_policy: Literal['soft', 'normal', 'hard'] = mutation_policy
        # -- Almaceno en una propiedad las restricciones a aplicar
        self.problem_restrictions: Literal['bound_restricted', 'full_restricted'] = problem_restrictions
        self.bounds_dict = self.individual_list[0].bounds_dict
        self.num_generations: int = num_generations

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
        total_generations = self.num_generations
        current_generation: int = individual.generation
        mutation_progress = 1 - (current_generation / total_generations)  # Reduce progresivamente

        # Definir los límites de mutación en función de la política
        match self.mutation_policy:
            case 'soft':
                # Inicialmente toma el 40% del rango total, reduciéndose con la generación
                percentage_range = 0.4 * mutation_progress
            case 'normal':
                # Inicialmente toma el 60% del rango total, reduciéndose con la generación
                percentage_range = 0.6 * mutation_progress
            case 'hard' | _:
                # Hard mantiene el rango completo sin reducción progresiva
                percentage_range = 0.8 * mutation_progress

        # Obtener los valores dentro de los límites permitidos
        possible_values = sorted(self.bounds_dict[parameter]["malformation_limits"])
        current_value = individual.get_individual_values()[parameter]

        # Determinar el rango basado en la generación actual
        min_val, max_val = min(possible_values), max(possible_values)
        range_span = max_val - min_val
        lower_bound = max(min_val, current_value - (range_span * percentage_range))
        upper_bound = min(max_val, current_value + (range_span * percentage_range))

        # Filtrar valores dentro del rango calculado
        filtered_values = [z for z in possible_values if lower_bound <= z <= upper_bound and z != current_value]

        if not filtered_values:  # Si no quedan valores posibles, tomar el más cercano válido
            filtered_values = [z for z in possible_values if z != current_value]

        return float(np.random.choice(filtered_values)) if self.bounds_dict[parameter]["type"] == "float" else int(np.random.choice(filtered_values))

    def mutation_uniform_old(self, individual, parameter):
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

    def mutation_uniform(self, individual, parameter):
        """
        Realiza una mutación uniforme en valores enteros o reales.

        :param individual: Individuo que se quiere mutar en alguno de sus genes.
        :param parameter: Parámetro que se quiere modificar del individuo.

        :return: Parámetro mutado.
        """
        total_generations = self.num_generations
        current_generation: int = individual.generation
        mutation_progress = 1 - (current_generation / total_generations)  # Reduce progresivamente

        # Definir los límites de mutación en función de la política
        match self.mutation_policy:
            case 'soft':
                percentage_range = 0.4 * mutation_progress
            case 'normal':
                percentage_range = 0.6 * mutation_progress
            case 'hard' | _:
                percentage_range = 0.8 * mutation_progress

        # Obtener los valores dentro de los límites permitidos
        possible_values = sorted(self.bounds_dict[parameter]["malformation_limits"])
        current_value = individual.get_individual_values()[parameter]

        # Determinar el rango basado en la generación actual
        min_val, max_val = min(possible_values), max(possible_values)
        range_span = max_val - min_val
        lower_bound = max(min_val, current_value - (range_span * percentage_range))
        upper_bound = min(max_val, current_value + (range_span * percentage_range))

        # Ajustar los límites para valores flotantes o enteros
        if self.bounds_dict[parameter]["type"] == "float":
            return float(np.random.uniform(lower_bound, upper_bound))
        else:  # "int"
            return int(np.random.uniform(np.floor(lower_bound), np.ceil(upper_bound)))

    # </editor-fold>
