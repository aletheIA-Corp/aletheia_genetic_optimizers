import random
from info_tools import InfoTools
from individuals import Individual
from typing import List
import numpy as np


class Reproduction:
    def __init__(self, winners_list: List[Individual], number_of_children: int, verbose: bool = True):
        self.winners_list: List[Individual] = winners_list
        self.number_of_children: int = number_of_children
        self.children_list: List[Individual] = []
        self.parents_generation: int = self.winners_list[0].generation
        self.verbose: bool = verbose
        self.IT: InfoTools = InfoTools()

    def run_reproduction(self):
        if self.verbose:
            self.IT.intro_print(f"RUN REPRODUCTION generacion: {self.parents_generation} -> sacará la generación {self.parents_generation+1}")
        # -- Obtengo el bounds dict fijo para que cada tipo de parametro tenga su propio cruce (rangos con cx_blend, fijos con cx_uniform)
        bounds_dict = self.winners_list[0].bounds_dict

        # -- Primero, iteramos hasta que la children_list tenga number_of_children individuos
        while len(self.children_list) < self.number_of_children:
            for individual in self.winners_list:

                if len(self.children_list) == self.number_of_children:
                    break
                elif len(self.children_list) > self.number_of_children:
                    self.children_list.pop(-1)
                    break

                # Seleccionamos otro individuo al azar con el que se va a cruzar
                random_individual_selected: Individual = random.choice([ind for ind in self.winners_list if ind != individual])
                child_one_values_list: list = []
                child_two_values_list: list = []

                if self.verbose:
                    self.IT.sub_intro_rint(f"Padres que se van a reproducir:")
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
                child_individual_1: Individual = Individual(bounds_dict, child_one_values_list, self.parents_generation+1)
                child_individual_2: Individual = Individual(bounds_dict, child_two_values_list, self.parents_generation+1)

                if self.verbose:
                    self.IT.sub_intro_rint(f"Hijos resultantes:")
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

