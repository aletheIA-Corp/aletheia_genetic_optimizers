import random
from info_tools import InfoTools
from aletheia_genetic_optimizers.individuals import Individual
from typing import List, Literal
import numpy as np


# TODO: Clase abstracta reproduction con varias clases hijas cada una para los diferentes tipos de reproduccion
# TODO:  ¿Lo hacemos dinamico a partir de un problem type o se debe escoger a manopla el metodo de reproduccio?

class Reproduction:
    def __init__(self, winners_list: List[Individual], number_of_children: int, problem_restrictions: str, problem_type: Literal["minimize", "maximize"], verbose: bool = True):
        self.winners_list: List[Individual] = winners_list
        self.number_of_children: int = number_of_children
        # -- Almaceno en una propiedad las restricciones a aplicar
        self.problem_restrictions: Literal['bound_restricted', 'full_restricted'] = problem_restrictions
        self.children_list: List[Individual] = []
        self.parents_generation: int = self.winners_list[0].generation
        self.problem_type: Literal["minimize", "maximize"] = problem_type
        self.verbose: bool = verbose
        self.IT: InfoTools = InfoTools()

    def run_reproduction(self):
        if self.verbose:
            self.IT.intro_print(f"RUN REPRODUCTION generacion: {self.parents_generation} -> sacará la generación {self.parents_generation + 1}")

        # -- Primero voy a agregar a los 10 % de mejores padres no repetidos a la children list
        num_to_select = max(1, int(len(self.winners_list) * 0.15))
        self.children_list += sorted(self.winners_list, key=lambda ind: ind.individual_fitness, reverse=self.problem_type == "maximize")[:num_to_select]

        # -- Despues, creo individuos idénticos a esos padres porque no puedo modificar su generación sin dejar descalza la generacion 0
        self.children_list = [Individual(ind.bounds_dict, ind.get_child_values(), self.parents_generation + 1, self.problem_restrictions) for ind in self.children_list]

        match self.problem_restrictions:
            case "bound_restricted":
                return self.bound_restricted_reproduction()
            case "full_restricted":
                return self.full_restricted_reproduction()

    def bound_restricted_reproduction(self):
        # -- Obtengo el bounds dict fijo para que cada tipo de parametro tenga su propio cruce (rangos con cx_blend, fijos con cx_uniform)
        bounds_dict = self.winners_list[0].bounds_dict

        # -- Pongo una cantidad fija de iteraciones de seguridad
        max_attempts: int = 4
        attempts: int = 0

        # -- Primero, iteramos hasta que la children_list tenga number_of_children individuos
        while len(self.children_list) < self.number_of_children:

            if attempts > max_attempts:
                break

            for individual in self.winners_list:

                if len(self.children_list) >= self.number_of_children:
                    break

                # Seleccionamos otro individuo al azar con el que se va a cruzar
                random_individual_selected: Individual = random.choice([ind for ind in self.winners_list if ind != individual])
                child_one_values_list: list = []
                child_two_values_list: list = []

                if self.verbose:
                    self.IT.sub_intro_print(f"Padres que se van a reproducir:")
                    self.IT.info_print(f"Padre 1: {individual.get_individual_values()}")
                    self.IT.info_print(f"Padre 2: {random_individual_selected.get_individual_values()}")

                # Realizamos los cruces de cada gen
                for parameter, bound in bounds_dict.items():
                    match bound['bound_type']:
                        case 'predefined':
                            c1, c2 = self.cx_uniform(individual.get_individual_values()[parameter], random_individual_selected.get_individual_values()[parameter])
                            child_one_values_list.append(c1)
                            child_two_values_list.append(c2)
                        case 'interval':
                            c1, c2 = self.cx_blend(individual.get_individual_values()[parameter], random_individual_selected.get_individual_values()[parameter])
                            child_one_values_list.append(c1 if bound["type"] == "float" else int(c1))
                            child_two_values_list.append(c2 if bound["type"] == "float" else int(c2))

                # Creamos los individuos y los agregamos a la lista si no existe ya uno similar
                child_individual_1: Individual = Individual(bounds_dict, child_one_values_list, self.parents_generation + 1, self.problem_restrictions)
                child_individual_2: Individual = Individual(bounds_dict, child_two_values_list, self.parents_generation + 1, self.problem_restrictions)

                if self.verbose:
                    self.IT.sub_intro_print(f"Hijos resultantes:")
                    self.IT.info_print(f"Hijo 1: {child_individual_1.get_individual_values()}")
                    self.IT.info_print(f"Hijo 2: {child_individual_2.get_individual_values()}")

                # Validamos que no existan individuos muy similares en la lista y que no tengan malformacion
                for ind in [child_individual_1, child_individual_2]:
                    is_duplicate = any(existing_indv == ind for existing_indv in self.children_list)
                    if not is_duplicate and not child_individual_1.malformation:
                        self.children_list.append(ind)

            attempts += 1

        return self.children_list

    def full_restricted_reproduction(self):
        # -- Obtengo el bounds dict fijo para que cada tipo de parametro tenga su propio cruce (rangos con cx_blend, fijos con cx_uniform)
        bounds_dict = self.winners_list[0].bounds_dict

        # -- Primero, iteramos hasta que la children_list tenga number_of_children individuos
        while len(self.children_list) < self.number_of_children:
            for individual in self.winners_list:

                if len(self.children_list) >= self.number_of_children:
                    break

                # Seleccionamos otro individuo al azar con el que se va a cruzar
                random_individual_selected: Individual = random.choice([ind for ind in self.winners_list if ind != individual])

                if self.verbose:
                    self.IT.sub_intro_print(f"Padres que se van a reproducir:")
                    self.IT.info_print(f"Padre 1: {individual.get_individual_values()}")
                    self.IT.info_print(f"Padre 2: {random_individual_selected.get_individual_values()}")

                # Realizamos los cruces entre dos padres para sacar un hijo
                c1 = self.ox1([z for z in individual.get_individual_values().values()], [z for z in random_individual_selected.get_individual_values().values()])
                c2 = self.ox1([z for z in individual.get_individual_values().values()], [z for z in random_individual_selected.get_individual_values().values()])

                # Creamos los individuos y los agregamos a la lista si no existe ya uno similar
                child_individual_1: Individual = Individual(bounds_dict, c1, self.parents_generation + 1, self.problem_restrictions)
                child_individual_2: Individual = Individual(bounds_dict, c2, self.parents_generation + 1, self.problem_restrictions)

                if self.verbose:
                    self.IT.sub_intro_print(f"Hijos resultantes:")
                    self.IT.info_print(f"Hijo 1: {child_individual_1.get_individual_values()}")
                    self.IT.info_print(f"Hijo 2: {child_individual_2.get_individual_values()}")

                # Validamos que no existan individuos muy similares en la lista y que no tengan malformacion
                for ind in [child_individual_1, child_individual_2]:
                    is_duplicate = any(existing_indv == ind for existing_indv in self.children_list)
                    if not is_duplicate and not child_individual_1.malformation:
                        self.children_list.append(ind)

        return self.children_list

    @staticmethod
    def cx_uniform(parent1, parent2, indpb=0.5):
        """Cruce uniforme con probabilidad indpb para valores individuales."""
        if np.random.rand() < indpb:
            return parent2, parent1  # Intercambio
        return parent1, parent2  # Sin cambio

    @staticmethod
    def cx_blend(parent1, parent2, alpha=0.5):
        """Cruce blend para valores continuos con dos padres individuales."""
        diff = abs(parent1 - parent2)
        low, high = min(parent1, parent2) - alpha * diff, max(parent1, parent2) + alpha * diff
        return np.random.uniform(low, high), np.random.uniform(low, high)

    @staticmethod
    def ox1(parent1, parent2):
        """Aplica Order Crossover (OX1) a dos padres de un problema basado en permutaciones."""

        # -- Creamos una lista del tamaño de los genes y la inicializamos en None
        size = len(parent1)
        offspring = [None] * size

        # Seleccionamos un segmento aleatorio en el rango permitido, por ejemplo de [0,1,2,3,4,5] se selecciona [2,3,4]
        start, end = sorted(random.sample(range(size), 2))

        # Copiar segmento del primer padre
        offspring[start:end + 1] = parent1[start:end + 1]

        # Completar con genes del segundo padre en el mismo orden, evitando repeticiones
        p2_genes = [gene for gene in parent2 if gene not in offspring]

        # Rellenar los espacios vacíos manteniendo el orden
        idx = 0
        for i in range(size):
            if offspring[i] is None:
                offspring[i] = p2_genes[idx]
                idx += 1

        return offspring
