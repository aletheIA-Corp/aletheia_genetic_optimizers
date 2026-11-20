from typing import Callable, List, Literal
from joblib import Parallel, delayed, parallel_config
from aletheia_genetic_optimizers.individuals import Individual


class Evaluation:
    """
    Clase encargada de evaluar individuos y asignar sus fitness en el proceso principal.
    Reutiliza los trabajadores entre generaciones mediante un contexto de ejecución.
    """

    def __init__(self, objective_function: Callable,
                 evaluation_method: Literal['sequential', 'process'] = 'process',
                 evaluation_workers: int = 4):
        """
        :param objective_function: Función que recibe un Individual y devuelve su fitness.
        :param evaluation_method: Evaluación secuencial o mediante procesos independientes.
        :param evaluation_workers: Número máximo de trabajadores simultáneos, mayor que cero.
        """

        # -- Validamos la configuración antes de crear los trabajadores
        if not callable(objective_function):
            raise ValueError("objective_function: Debe ser una función invocable.")
        if evaluation_method not in ('sequential', 'process'):
            raise ValueError("evaluation_method: Debe ser 'sequential' o 'process'.")
        if isinstance(evaluation_workers, bool) or not isinstance(evaluation_workers, int) or evaluation_workers < 1:
            raise ValueError("evaluation_workers: Debe ser un entero mayor que cero.")

        # -- Almacenamos la función y la configuración de evaluación
        self.objective_function: Callable = objective_function
        self.evaluation_method: Literal['sequential', 'process'] = evaluation_method
        self.evaluation_workers: int = evaluation_workers
        self._parallel: Parallel | None = None

    def __enter__(self) -> 'Evaluation':
        """Inicializa el contexto que mantiene los trabajadores entre generaciones."""

        # -- Loky permite serializar funciones internas y variables capturadas
        if self.evaluation_method == 'process':
            with parallel_config(backend='loky', inner_max_num_threads=1):
                self._parallel = Parallel(n_jobs=self.evaluation_workers)
                self._parallel.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Libera el contexto de trabajadores incluso si falla una evaluación."""
        if self._parallel is not None:
            try:
                self._parallel.__exit__(exc_type, exc_value, traceback)
            finally:
                self._parallel = None

    def run_evaluation(self, individual_list: List[Individual]) -> None:
        """
        Evalúa los individuos válidos y asigna los resultados conservando su orden.
        En modo procesos, los cambios realizados por la función en sus copias no se recuperan.
        :param individual_list: Individuos originales de la generación a evaluar.
        :return: None. Los fitness se asignan en los individuos originales.
        """

        # -- Excluimos las malformaciones antes de enviar tareas a los trabajadores
        valid_individuals = [ind for ind in individual_list if not ind.malformation]
        if not valid_individuals:
            return

        # -- Esperamos todos los resultados antes de actualizar la generación
        if self.evaluation_method == 'sequential':
            fitness_values = [self.objective_function(ind) for ind in valid_individuals]
        else:
            if self._parallel is None:
                raise RuntimeError("Evaluation: El modo procesos requiere utilizar 'with Evaluation(...)'.")
            fitness_values = self._parallel(delayed(self.objective_function)(ind) for ind in valid_individuals)

        # -- La asignación siempre ocurre en los individuos del proceso principal
        for individual, fitness in zip(valid_individuals, fitness_values):
            individual.individual_fitness = fitness
