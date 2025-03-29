import pandas as pd
import numpy as np
from typing import Literal
from abc import ABC, abstractmethod
from info_tools import InfoTools


class VariabilityExplossion(ABC):
    def __init__(self, early_stopping_generations: int, problem_type: Literal['maximize', 'minimize'], round_decimals: int = 3, verbose: bool = False):

        # -- Obtengo las generaciones que voy a esperar para que si se repite la moda, arrancar la explosion de variabilidad
        self.early_stopping_generations: int = early_stopping_generations

        # -- almaceno el tipo de problema e instancio InfoTools
        self.problem_type: Literal['maximize', 'minimize'] = problem_type
        self.IT: InfoTools = InfoTools()
        self.round_decimals: int = round_decimals
        self.verbose: int = verbose

        # -- Inicializo propiedades de control de flujo
        self.early_stopping_generations_executed_counter: int = 0  # Contador de cuantas generacioes han transcurrido desde la ultima explosion de variabilidad
        self.total_early_stopping_generations_executed_counter: int = 0  # Contador de cuantas generacioes han transcurrido desde la primera explosion de variabilidad
        self.early_stopping_generations_executed: bool = False  # Boleana que indica si estamos en modo explosion de variabilidad

        # -- Inicializo propiedades de mutacion (valores por defecto)
        self.mutate_probability: float = 0.4
        self.mutate_gen_probability: float = 0.25
        self.mutation_policy: Literal['soft', 'normal', 'hard'] = 'soft'

    @abstractmethod
    def evaluate_early_stopping(self, generations_fitness_statistics_df: pd.DataFrame | None) -> None:
        pass

    @abstractmethod
    def execute_variability_explossion(self):
        pass

    @abstractmethod
    def stop_genetic_iterations(self, generations_fitness_statistics_df: pd.DataFrame | None):
        pass

    @abstractmethod
    def print_variability_status(self):
        pass


class CrazyVariabilityExplossion(VariabilityExplossion):
    def __init__(self, early_stopping_generations: int, problem_type: Literal['maximize', 'minimize'], round_decimals: int = 3, verbose: bool = False):

        # -- Obtengo las generaciones que voy a esperar para que si se repite la moda, arrancar la explosion de variabilidad
        super().__init__(early_stopping_generations, problem_type, round_decimals, verbose)

    def evaluate_early_stopping(self, generations_fitness_statistics_df: pd.DataFrame | None) -> tuple:
        """
        Método para evaluar si se han cumplido las condiciones del early stopping
        :return:
        """

        # -- Si en las últimas early_stopping_generations el max es igual, aplicamos la explosión de variabilidad
        if generations_fitness_statistics_df is not None:
            df: pd.DataFrame = generations_fitness_statistics_df
            df_tail = df.tail(self.early_stopping_generations)
            if df.shape[0] >= int(self.early_stopping_generations * 2):
                mode_values = df_tail["mode"].values  # Extraer los valores como array
                if np.all(mode_values == mode_values[0]):
                    return self.execute_variability_explossion()

        # -- Si ya se ha ejecutado la explosion de variabilidad, se vuelve a ejecutar en cada iteracion
        if self.early_stopping_generations_executed:
            return self.execute_variability_explossion()

        # -- Si no ha pasado Nada, retornamos None
        return None, None, None, None

    def execute_variability_explossion(self):
        """
        Método para ejecutar una explosion de variabilidad en caso de que el genético haya quedado clavado en un mínimo. Esto pretende dar una oportunidad extra al modelo genético
        de encontrar un mínimo mejor cuando ha quedado atrapado en un mínimo local.
        :return:
        """

        if self.early_stopping_generations_executed:
            self.mutate_probability: float = 0.4
            self.mutate_gen_probability: float = 0.25
            self.mutation_policy: Literal['soft', 'normal', 'hard'] = 'soft'
        else:
            self.mutate_probability: float = 0.9
            self.mutate_gen_probability: float = 0.5
            self.mutation_policy: Literal['soft', 'normal', 'hard'] = 'hard'
            self.early_stopping_generations_executed = True

        return self.mutate_probability, self.mutate_gen_probability, self.mutation_policy, self.early_stopping_generations_executed

    def stop_genetic_iterations(self, generations_fitness_statistics_df: pd.DataFrame | None):
        """
        Método para que en caso de que si se ha ejecutado el execute_variability_explossion (early_stopping_generations_executed = True), y hayan pasado n generaciones, se detiene.
        :return:
        """
        if self.early_stopping_generations_executed_counter >= self.early_stopping_generations:
            # Si no ha mejorado en las vueltas, True, sino reseteamos el contador para darle margen
            df: pd.DataFrame = generations_fitness_statistics_df
            best_value = max(df["max"].values) if self.problem_type == "maximize" else min(df["max"].values)
            best_counter_value = max(df.tail(self.early_stopping_generations_executed_counter)["max"].values) if self.problem_type == "maximize" else (
                min(df.tail(self.early_stopping_generations_executed_counter)["max"].values))

            if self.problem_type == "maximize":
                if best_counter_value <= best_value:
                    self.IT.info_print("El CRAZY MODE no ha conseguido mejorar el resultado. Detenemos el proceso de evolución", "light_yellow")
                    return True
                else:
                    self.IT.info_print("El CRAZY MODE ha mejorado el mejor resultado. Le damos margen de generaciones", "light_blue")
                    self.early_stopping_generations_executed_counter = 0
                    return False
            else:
                if best_counter_value >= best_value:
                    self.IT.info_print("El CRAZY MODE no ha conseguido mejorar el resultado. Detenemos el proceso de evolución", "light_yellow")
                    return True
                else:
                    self.IT.info_print("El CRAZY MODE ha mejorado el mejor resultado. Le damos margen de generaciones", "light_blue")
                    self.early_stopping_generations_executed_counter = 0
                    return False

        if self.early_stopping_generations_executed:
            self.early_stopping_generations_executed_counter += 1
            self.total_early_stopping_generations_executed_counter += 1

        # -- En caso de verbose activo, mostramos la info
        if self.verbose:
            self.print_variability_status()

        return False

    def print_variability_status(self):
        self.IT.sub_intro_rint(f"Resumen de CrazyVariabilityExplossion")
        self.IT.info_print(f"CrazyVariabilityExplossion Activated: {self.early_stopping_generations_executed}",
                           "light_red" if self.early_stopping_generations_executed else "light_green")
        if self.early_stopping_generations_executed:
            self.IT.info_print(f"Generaciones que lleva activo el CrazyVariabilityExplossion: {self.total_early_stopping_generations_executed_counter}")
            self.IT.info_print(f"Generaciones desde ultima mejora: {self.early_stopping_generations_executed_counter}")