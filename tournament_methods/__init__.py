from abc import ABC, abstractmethod
from individuals import Individual
from typing import List, Literal


class Tournament(ABC):
    def __init__(self, podium_size: int = 3, problem_type: Literal["minimize", "maximize"] = "minimize"):
        self.podium_size: int = podium_size
        self.problem_type: Literal["minimize", "maximize"] = problem_type

    @abstractmethod
    def run_tournament(self, individuals_list: List[Individual]) -> List[Individual]:
        pass


class EaSimple(Tournament):
    def __init__(self, podium_size: int = 3, problem_type: Literal["minimize", "maximize"] = "minimize"):
        super().__init__(podium_size, problem_type)

    def run_tournament(self, individuals_list: List[Individual]) -> List[List]:
        # -- Primero, eliminamos de la lista todos los individuos con malformacion
        individuals_list = [ind for ind in individuals_list if not ind.malformation]

        # -- Ejecutamos el torneo y retornamos una lista de listas con el siguiente formato

        # TODO: De momento para jugar voy a retornar los 5 mejores, pero hay que armar la lógica de competicion
        return sorted(individuals_list, key=lambda ind: ind.individual_fitness, reverse=True if self.problem_type == "maximize" else False)[0:5]


