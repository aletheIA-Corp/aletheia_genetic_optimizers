from typing import Callable, Dict
from aletheia_genetic_optimizers.tournament_methods import *
from aletheia_genetic_optimizers.individuals import Individual
from aletheia_genetic_optimizers.reproduction_methods import Reproduction
from aletheia_genetic_optimizers.mutation_methods import Mutation
from aletheia_genetic_optimizers.population_methods import Population


class GenethicOptimizer:
    def __init__(self,
                 bounds_dict: Dict,
                 num_generations: int,
                 num_individuals: int,
                 objective_function: Callable,
                 problem_restrictions: Literal['bound_restricted', 'full_restricted'] = "bound_restricted",
                 problem_type: Literal["minimize", "maximize"] = "minimize",
                 tournament_method: Literal["ea_simple"] = "ea_simple",
                 podium_size: int = 3,
                 mutate_probability: float = 0.25,
                 mutate_gen_probability: float = 0.2,
                 mutation_policy: Literal['soft', 'normal', 'hard'] = 'medium',
                 verbose: bool = True
                 ):
        """
        Clase-Objeto padre para crear un algoritmo genético cuántico basado en QAOA y generacion de aleatoriedad cuántica
        en lo respectivo a mutaciones y cruces reproductivos.
        :param bounds_dict: Diccionario en el que se definen los parámetros a optimizar y sus valores, ej. '{learning_rate: (0.0001, 0.1)}'
        :param num_generations: Numero de generaciones que se van a ejecutar
        :param num_individuals: Numero de Individuos iniciales que se van a generar
        :param objective_function: Función objetivo que se va a emplear para puntuar a cada individuo (debe retornar un float)
        :param problem_restrictions: ['bound_restricted', 'full_restricted'] Restricciones que se van a aplicar a la hora de crear individuos, reprocirlos y mutarlos
        :param problem_type: [minimize, maximize] Seleccionar si se quiere minimizar o maximizar el resultado de la función objetivo. Por ejemplo si usamos un MAE es minimizar,
         un Accuracy sería maximizar.
        :param tournament_method: [easimple, .....] Elegir el tipo de torneo para seleccionar los individuos que se van a reproducir.
        :param podium_size: Cantidad de individuos de la muestra que van a competir para elegir al mejor. Por ejemplo, si el valor es 3, se escogen iterativamente 3 individuos
        al azar y se selecciona al mejor. Este proceso finaliza cuando ya no quedan más individuos y todos han sido seleccionados o deshechados.
        :param mutate_probability:Tambien conocido como indpb ∈[0, 1]. Probabilidad de mutar que tiene cada gen. Una probabilidad de 0, implica que nunca hay mutación,
        una probabilidad de 1 implica que siempre hay mutacion.
        """
        # -- Almaceno propiedades
        self.bounds_dict: Dict = bounds_dict
        self.num_generations: int = num_generations
        self.num_individuals: int = num_individuals
        self.objective_function: Callable = objective_function
        self.problem_restrictions: Literal['bound_restricted', 'full_restricted'] = problem_restrictions
        self.problem_type: str = problem_type
        self.tournament_method: str = tournament_method
        self.podium_size: int = podium_size
        self.mutate_probability: float = mutate_probability
        self.mutate_gen_probability: float = mutate_gen_probability
        self.mutation_policy: Literal['soft', 'normal', 'hard'] = mutation_policy
        self.verbose: bool = verbose

        # -- instancio info tools para los prints
        self.IT: InfoTools = InfoTools()

        # -- Instancio la clase GenethicTournamentMethods en GTM y almaceno el torneo
        self.GTM: Tournament = self.get_tournament_method(self.verbose)

        # -- Almaceno cualquiera de los bounds_dict en self.bounds_dict y modifico self.predefined_bounds_problem
        if self.bounds_dict is None:
            raise ValueError("Se requiere uno de estos dos parámetros: bounds_dict_random, bounds_dict_predefined. Ambos han sido rellenados con None")

        if self.verbose:
            self.IT.sub_intro_rint(f"Bounds_dict y valores a combinar")
            for k, v in self.bounds_dict.items():
                self.IT.info_print(f"{k}: {v}")

        # -- Validamos los inputs
        self.validate_input_parameters()

        # -- Creamos el objeto poblacion y la poblacion inicial
        self.POPULATION: Population = Population(self.bounds_dict, self.num_individuals, self.problem_restrictions)

        # -- Creamos las listas de individuos que vamos a ir usando
        self.POPULATION.create_population()

        # -- Pasamos a cada individuo de la generacion 0 por la funcion de coste
        for individual in self.POPULATION.populuation_dict[0]:
            individual.individual_fitness = self.objective_function(individual)

        if self.verbose:
            self.print_generation_info(self.POPULATION.populuation_dict[0], 0)

        # -- Entramos a la parte genetica iterando por generaciones
        for gen in range(1, self.num_generations):

            # -- Ejecutamos el torneo para obtener los padres ganadores en base a los individuos de la generacion anterior
            winners_list: List[Individual] = self.GTM.run_tournament(self.POPULATION.populuation_dict[gen - 1])

            # -- Creamos los hijos y los agregamos a la lista de individuos
            children_list: List[Individual] = Reproduction(winners_list, self.num_individuals, self.problem_restrictions, self.problem_type,False).run_reproduction()

            # -- Mutamos los individuos
            children_list = Mutation(children_list, self.mutate_probability, self.mutate_gen_probability, self.mutation_policy, self.problem_restrictions).run_mutation()

            # -- Agregamos los individuos al diccionario de poblacion en su generacion correspondiente
            self.POPULATION.add_generation_population(children_list, gen)

            # -- Pasamos a cada individuo por la funcion de coste
            for individual in self.POPULATION.populuation_dict[gen]:
                if not individual.malformation:
                    individual.individual_fitness = self.objective_function(individual)

            if self.verbose:
                self.print_generation_info(self.POPULATION.populuation_dict[gen], gen)

                best = None
                for gen, ind_list in self.POPULATION.populuation_dict.items():
                    best_gen_ind = sorted(self.POPULATION.populuation_dict[gen], key=lambda ind: ind.individual_fitness, reverse=self.problem_type == 'maximize')[0]
                    self.IT.info_print(f"Mejor ind gen: {gen}: {best_gen_ind.get_individual_values()} - Fitness: {best_gen_ind.individual_fitness}")

                    if self.problem_type == "maximize":
                        if best is None:
                            best = best_gen_ind
                        else:
                            if best.individual_fitness < best_gen_ind.individual_fitness:
                                best = best_gen_ind
                    else:
                        if best is None:
                            best = best_gen_ind
                        else:
                            if best.individual_fitness > best_gen_ind.individual_fitness:
                                best = best_gen_ind

                self.IT.info_print(f"Mejor ind TOTAL: {gen}: {best.get_individual_values()} - Fitness: {best.individual_fitness}")

    def validate_input_parameters(self) -> bool:
        """
        Método para validar los inputs que se han cargado en el constructor
        :return: True si todas las validaciones son correctas Excepction else
        """

        # -- Validar el bounds_dict en cada caso (interval y predefined)

        # INTERVAL
        interval_bounds_dict: dict = {k: v for k, v in self.bounds_dict.items() if v["bound_type"] == "interval"}
        if not all(isinstance(valor, (int, float)) for param in interval_bounds_dict for key in ["limits", "malformation_limits"] if key in param for valor in param[key]):
            raise ValueError("bounds_dict: No todos los valores en los bounds_dict interval son int o float.")

        # PREDEFINED
        predefined_bounds_dict: dict = {k: v for k, v in self.bounds_dict.items() if v["bound_type"] == "predefined"}
        if not all(isinstance(valor, (int, float)) for param in predefined_bounds_dict for key in ["limits", "malformation_limits"] if key in param for valor in param[key]):
            raise ValueError("bounds_dict: No todos los valores en los bounds_dict interval son int o float.")

        # -- Validar Enteros num_generations, num_individuals, podium_size
        if not isinstance(self.num_generations, int):
            raise ValueError(f"self.num_generations: Debe ser un entero y su tipo es {type(self.num_generations)}")
        if not isinstance(self.num_individuals, int):
            raise ValueError(f"self.num_individuals: Debe ser un entero y su tipo es {type(self.num_individuals)}")
        if not isinstance(self.podium_size, int):
            raise ValueError(f"self.podium_size: Debe ser un entero y su tipo es {type(self.podium_size)}")

        # -- Validar Flotantes mutate_probability
        if not isinstance(self.mutate_probability, float):
            raise ValueError(f"self.mutate_probability: Debe ser un float y su tipo es {type(self.mutate_probability)}")

        # -- Validar strings problem_type, tournament_method
        if not isinstance(self.problem_type, str):
            raise ValueError(f"self.problem_type: Debe ser un str y su tipo es {type(self.problem_type)}")
        if self.problem_type not in ["minimize", "maximize"]:
            raise ValueError(f'self.problem_type debe ser una opción de estas: ["minimize", "maximize"] y se ha pasado {self.problem_type}')
        if not isinstance(self.tournament_method, str):
            raise ValueError(f"self.tournament_method: Debe ser un str y su tipo es {type(self.tournament_method)}")

        return True

    def get_tournament_method(self, verbose):
        """
        Método que crea y retorna el tournament seleccionado
        :return:
        """
        match self.tournament_method:
            case "ea_simple":
                return EaSimple(self.podium_size, self.problem_type, verbose)

    def print_generation_info(self, individual_generation_list: List[Individual], generation: int):
        self.IT.intro_print(f"Individuos generacion {generation}")
        self.IT.sub_intro_rint("Información de los individuos y los fitness")
        for i, ind in enumerate([z for z in individual_generation_list if z.generation == generation]):
            pad_number = lambda num: str(num).zfill(len(str(self.num_individuals)))
            self.IT.info_print(f"Individuo {pad_number(i + 1)}: {ind.get_individual_values()} - Generación: {ind.generation} - [Fitness]: {ind.individual_fitness}")

        self.IT.sub_intro_rint(f"Información de la evolución de las distribuciones en cada generación")
        self.IT.print_tabulate_df(self.POPULATION.get_generation_fitness_statistics(generation), row_print=self.num_generations+1)

    def plot_generation_stats(self) -> None:
        self.POPULATION.plot_generation_stats()

    def plot_evolution_animated(self) -> None:
        self.POPULATION.plot_evolution_animated()

    def plot_evolution(self) -> None:
        self.POPULATION.plot_evolution()
