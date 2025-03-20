from info_tools import InfoTools
from pandas_tools import PandasTools
from typing import Callable, Literal, Dict, Tuple, Union, List
from tournament_methods import *
from bounds import BoundRandomCreator, BoundPredefinedCreator
from individuals import Individual


class GenethicOptimizer:
    def __init__(self,
                 bounds_dict_random: Dict[str, Tuple[Union[int, float]]] | None,
                 bounds_dict_predefined: Dict[str, Tuple[Union[int, float]]] | None,
                 num_generations: int,
                 num_individuals: int,
                 objective_function: Callable,
                 problem_type: Literal["minimize", "maximize"] = "minimize",
                 tournament_method: Literal["ea_simple"] = "ea_simple",
                 podium_size: int = 3,
                 reproduction_variability: float = 0.2,
                 mutate_probability: float = 0.25,
                 mutation_center_mean: float = 0.0,
                 mutation_size: float = 0.5,
                 verbose: bool = True
                 ):
        """
        Clase-Objeto padre para crear un algoritmo genético cuántico basado en QAOA y generacion de aleatoriedad cuántica
        en lo respectivo a mutaciones y cruces reproductivos.
        :param bounds_dict_random: Diccionario en el que se definen los parámetros a optimizar y sus valores, ej. '{learning_rate: (0.0001, 0.1)}'
        Se utiliza cuando los valores pueden oscilar en un rango sin romper la lógica de negocio.
        :param bounds_dict_predefined: Diccionario en el que se definen los parámetros a optimizar y sus valores, ej. '{swich_1: (-1, 0, 1), swich_2: (-1, 0, 1)}'
        Se utiliza cuando los valores a optimizar están predefinidos y no pueden variar. En este caso los individuos no pueden tener valores diferentes de los predefinidos.
        :param num_generations: Numero de generaciones que se van a ejecutar
        :param num_individuals: Numero de Individuos iniciales que se van a generar
        :param objective_function: Función objetivo que se va a emplear para puntuar a cada individuo (debe retornar un float)
        :param problem_type: [minimize, maximize] Seleccionar si se quiere minimizar o maximizar el resultado de la función objetivo. Por ejemplo si usamos un MAE es minimizar,
         un Accuracy sería maximizar.
        :param tournament_method: [easimple, .....] Elegir el tipo de torneo para seleccionar los individuos que se van a reproducir.
        :param podium_size: Cantidad de individuos de la muestra que van a competir para elegir al mejor. Por ejemplo, si el valor es 3, se escogen iterativamente 3 individuos
        al azar y se selecciona al mejor. Este proceso finaliza cuando ya no quedan más individuos y todos han sido seleccionados o deshechados.
        :param reproduction_variability: También conocido como Alpha. α∈[0,1]. Directamente proporcional a la potencial variablidad entre hijos y padres. Ej. Si Alpha=0, los genes
        de los hijos solo van a poder mutar en una interpolación entre los valores de los padres, cumpliendo la siguiente ecuación: λ∈[−α,1+α], para este caso λ∈[0,1]. Si Alpha=0.5
        λ∈[−0.5,1.5]. Esto se calculará posteriormente en una magnitud proporcional a los valores de los genes de los padres.
        :param mutate_probability:Tambien conocido como indpb ∈[0, 1]. Probabilidad de mutar que tiene cada gen. Una probabilidad de 0, implica que nunca hay mutación,
        una probabilidad de 1 implica que siempre hay mutacion.
        :param mutation_center_mean: μ donde μ∈R. Sesgo que aplicamos a la mutación para que el gen aumente o disminuya su valor. Cuando es 0, existe la misma probabilidad de
        mutar positiva y negativamente. Cuando > 0, aumenta proporcionalmente la probabilidad de mutar positivamente y viceversa. v=v0+N(μ,σ)
        :param mutation_size σ donde σ>0. Desviación estándar de la mutación, Si sigma es muy pequeño (ej. 0.01), las mutaciones serán mínimas, casi insignificantes.
        Si sigma es muy grande (ej. 10 o 100), las mutaciones pueden ser demasiado bruscas, afectando drásticamente la solución.
        - Mutaciones pequeñas, estables 0.1 - 0.5
        - Balance entre estabilidad y exploración 0.5 - 1.0
        - Exploración agresiva 1.5 - 3.0
        """
        # -- Almaceno propiedades
        self.bounds_dict_random: Dict[str, Tuple[Union[int, float]]] | None = bounds_dict_random
        self.bounds_dict_predefined: Dict[str, Tuple[Union[int, float]]] | None = bounds_dict_predefined
        self.predefined_bounds_problem: bool = False  # Defino esta boleana para marcar si estamos trabajando con bounds predefinidos o no
        self.num_generations: int = num_generations
        self.num_individuals: int = num_individuals
        self.objective_function: Callable = objective_function
        self.problem_type: str = problem_type
        self.tournament_method: str = tournament_method
        self.podium_size: int = podium_size
        self.reproduction_variability: float = reproduction_variability
        self.mutate_probability: float = mutate_probability
        self.mutation_center_mean: float = mutation_center_mean
        self.mutation_size: float = mutation_size
        self.mutation_size: float = mutation_size
        self.verbose: bool = verbose

        # -- Instancio la clase GenethicTournamentMethods en GTM
        self.GTM: Tournament = self.get_tournament_method()

        # -- Almaceno cualquiera de los bounds_dict en self.bounds_dict y modifico self.predefined_bounds_problem
        if self.bounds_dict_random is None and self.bounds_dict_predefined is None:
            raise ValueError("Se requiere uno de estos dos parámetros: bounds_dict_random, bounds_dict_predefined. Ambos han sido rellenados con None")
        if self.bounds_dict_random is not None and self.bounds_dict_predefined is not None:
            raise ValueError("QGO solo puede trabajar con uno de estos dos parametros: bounds_dict_random, bounds_dict_predefined. Ambos tienen valor, debes escoger uno")
        if self.bounds_dict_predefined is not None:
            self.predefined_bounds_problem = True
            self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = self.bounds_dict_predefined
        else:
            self.predefined_bounds_problem = False
            self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = self.bounds_dict_random

        if self.verbose:
            print(f"\n######################    Bounds_dict y valores a combinar     ######################\n")
            print(f"[INFO] ------> Los posibles valores a combinar son predefinidos: {self.predefined_bounds_problem}")
            print(f"[INFO] ------> bounds_dict: {self.bounds_dict}")

        # -- Validamos los inputs
        self.validate_input_parameters()

        # -- En base a si los bounds son predefinidos o no, Creamos los individuos y los almacenamos en una lista
        self.individuals_list: List[Individual] = []
        for i in range(self.num_individuals):
            self.individuals_list.append(Individual(self.bounds_dict, None, 0, self.predefined_bounds_problem))

        if self.verbose:
            print(f"\n######################    Individuos 1a generacion     ######################\n")
            for i, ind in enumerate(self.individuals_list):
                print(f"Individuo {i + 1}: {ind.get_individual_values()} - Generación: {ind.generation}")
        breakpoint()

        # -- TODO: Entramos a la parte genetica iterando por generaciones
        for gen in self.num_generations:

            # TODO: Ejecutamos el torneo para obtener los hijos y creamos la child_list

            # -- Se supone que ya hemos obtenido los hijos
            child_list: List[List] = [
                [0.08910494983403751, 36.5],
                [0.07170969052131343, 71.5],
                [0.005021695515233724, 16.5],
                [0.015734475416848345, 856.5],
            ]

            # -- Agregamos los individuos a la lista
            self.individuals_list += [Individual(self.bounds_dict, child_vals, gen, self.predefined_bounds_problem) for child_vals in child_list]

            for individual in [ind for ind in self.individuals_list if ind.generation == gen]:
                print(f"Malformation: {individual.malformation} - Values: {individual.get_individual_values()}")

            # self.individuals_list = Individuals(self.bounds_dict, self.num_individuals, False, child_list).get_individuals()

            print(self.individuals_list)

    def validate_input_parameters(self) -> bool:
        """
        Método para validar los inputs que se han cargado en el constructor
        :return: True si todas las validaciones son correctas Excepction else
        """

        # -- Validar el bounds_dict
        if not all(isinstance(valor, (int, float)) for param_data in self.bounds_dict.values()
                   for key in ["limits", "malformation_limits"] if key in param_data for valor in param_data[key]):
            raise ValueError("bounds_dict: No todos los valores en bounds_dict son int o float.")

        if not self.predefined_bounds_problem:
            if not all(len(values['limits']) == 2 for values in self.bounds_dict.values()):
                raise ValueError("bounds_dict: has seleccionado un bounds_dict aleatorio, pero has agregado mas elementos de un limite máximo y uno mínimo.")

        # -- Validar Enteros num_generations, num_individuals, podium_size
        if not isinstance(self.num_generations, int):
            raise ValueError(f"self.num_generations: Debe ser un entero y su tipo es {type(self.num_generations)}")
        if not isinstance(self.num_individuals, int):
            raise ValueError(f"self.num_individuals: Debe ser un entero y su tipo es {type(self.num_individuals)}")
        if not isinstance(self.podium_size, int):
            raise ValueError(f"self.podium_size: Debe ser un entero y su tipo es {type(self.podium_size)}")

        # -- Validar Flotantes reproduction_variability, mutate_probability, mutation_center_mean, mutation_size
        if not isinstance(self.reproduction_variability, float):
            raise ValueError(f"self.reproduction_variability: Debe ser un float y su tipo es {type(self.reproduction_variability)}")
        if not isinstance(self.mutate_probability, float):
            raise ValueError(f"self.mutate_probability: Debe ser un float y su tipo es {type(self.mutate_probability)}")
        if not isinstance(self.mutation_center_mean, float):
            raise ValueError(f"self.mutation_center_mean: Debe ser un float y su tipo es {type(self.mutation_center_mean)}")
        if not isinstance(self.mutation_size, float):
            raise ValueError(f"self.mutation_size: Debe ser un float y su tipo es {type(self.mutation_size)}")
        if self.mutation_size < 0:
            raise ValueError(f"self.mutation_size: Debe ser un float >= 0 y su valor es {self.mutation_size}")

        # -- Validar strings problem_type, tournament_method
        if not isinstance(self.problem_type, str):
            raise ValueError(f"self.problem_type: Debe ser un str y su tipo es {type(self.problem_type)}")
        if self.problem_type not in ["minimize", "maximize"]:
            raise ValueError(f'self.problem_type debe ser una opción de estas: ["minimize", "maximize"] y se ha pasado {self.problem_type}')
        if not isinstance(self.tournament_method, str):
            raise ValueError(f"self.tournament_method: Debe ser un str y su tipo es {type(self.tournament_method)}")

        return True

    def get_tournament_method(self):
        """
        Método que crea y retorna el tournament seleccionado
        :return:
        """
        match self.tournament_method:
            case "ea_simple":
                return EaSimple(self.podium_size)


def example_1_bounds_no_predefinidos():
    # -- Creamos el diccionario de bounds
    bounds = BoundRandomCreator()
    bounds.add_bound("learning_rate", 0.0001, 0.1, 0.000001, 1, "float")
    bounds.add_bound("batch_size", 12, 64, 8, 124, "int")

    print(GenethicOptimizer(bounds.get_bound(),
                            None,
                            5,
                            20,
                            lambda x: x + 1,
                            "minimize",
                            "ea_simple",
                            3,
                            0.2,
                            0.25,
                            0.0,
                            0.5,
                            ))


def example_2_bounds_predefinidos():
    # -- Creamos el diccionario de bounds
    bounds = BoundPredefinedCreator()
    bounds.add_bound("learning_rate", [0.1, 0.01, 0.001, 0.0001], "float")
    bounds.add_bound("batch_size", [16, 32, 64], "int")

    print(GenethicOptimizer(None,
                            bounds.get_bound(),
                            5,
                            20,
                            lambda x: x + 1,
                            "minimize",
                            "ea_simple",
                            3,
                            0.2,
                            0.25,
                            0.0,
                            0.5,

                            ))


example_1_bounds_no_predefinidos()
# example_2_bounds_predefinidos()
