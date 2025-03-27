from abc import ABC, abstractmethod
from aletheia_genetic_optimizers.individuals import Individual
from typing import List, Literal
import random
from info_tools import InfoTools


class Tournament(ABC):
    def __init__(self, podium_size: int = 3, problem_type: Literal["minimize", "maximize"] = "minimize", verbose: bool = False):
        self.podium_size: int = podium_size
        self.problem_type: Literal["minimize", "maximize"] = problem_type
        self.verbose: bool = verbose
        self.IT: InfoTools = InfoTools()

    @abstractmethod
    def run_tournament(self, individuals_list: List[Individual]) -> List[Individual]:
        pass


class EaSimple(Tournament):
    def __init__(self, podium_size: int = 3, problem_type: Literal["minimize", "maximize"] = "minimize", verbose: bool = False):
        super().__init__(podium_size, problem_type, verbose)

    def run_tournament(self, individuals_list: List[Individual]) -> List[Individual]:
        # Filtrar individuos con malformaciones
        individuals_list = [ind for ind in individuals_list if not ind.malformation]

        selected_individuals = []  # Lista de ganadores del torneo

        if self.verbose:
            self.IT.sub_intro_rint("Torneo EaSimple")

        while individuals_list:
            # Seleccionar hasta 3 individuos aleatoriamente
            competitors = random.sample(individuals_list, min(self.podium_size, len(individuals_list)))

            # Seleccionar el mejor según el tipo de problema
            winner = max(competitors, key=lambda ind: ind.individual_fitness) if self.problem_type == "maximize" else min(competitors, key=lambda ind: ind.individual_fitness)
            selected_individuals.append(winner)

            if self.verbose:
                self.IT.info_print(f"Competidores: {[z.individual_fitness for z in competitors]} - Winner: {winner.individual_fitness}", "light_magenta")

            # Eliminar todos los competidores de la lista (no solo el ganador)
            individuals_list = [ind for ind in individuals_list if ind not in competitors]

        return selected_individuals


