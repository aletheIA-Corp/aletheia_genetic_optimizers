from abc import ABC, abstractmethod
from individuals import Individual
from typing import List


class Tournament(ABC):
    def __init__(self, podium_size: int = 3):
        self.podium_size: int = podium_size

    @abstractmethod
    def run_tournament(self, individuals_list: List[Individual]):
        pass


class EaSimple(Tournament):
    def __init__(self, podium_size: int = 3):
        super().__init__(podium_size)

    def run_tournament(self, individuals_list: List[Individual]):
        pass