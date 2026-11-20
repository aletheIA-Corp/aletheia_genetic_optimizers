import contextlib
import io
import os
import random
import unittest

import numpy as np

from aletheia_genetic_optimizers import GenethicOptimizer, Individual
from aletheia_genetic_optimizers.evaluation_methods import Evaluation


BOUNDS = {'x': {'bound_type': 'interval', 'type': 'float', 'limits': (0, 100), 'malformation_limits': (0, 100)}}


def make_individual(value):
    return Individual(BOUNDS, [value], 0, 'bound_restricted')


class EvaluationTests(unittest.TestCase):
    def test_nested_function_order_and_malformations(self):
        offset = 7

        def objective(individual):
            return individual.get_individual_values()['x'] ** 2 + offset

        for method in ('sequential', 'process'):
            individuals = [make_individual(value) for value in (3, -1, 1, 2)]
            with Evaluation(objective, method) as evaluation:
                evaluation.run_evaluation(individuals)
                evaluation.run_evaluation([])
                evaluation.run_evaluation([individuals[1]])
            self.assertEqual([ind.individual_fitness for ind in individuals], [16, None, 8, 11])
            self.assertIsNone(evaluation._parallel)

    def test_process_execution_and_reuse(self):
        main_pid = os.getpid()
        with Evaluation(lambda ind: os.getpid()) as evaluation:
            first = [make_individual(i) for i in range(12)]
            second = [make_individual(i) for i in range(12)]
            executor = evaluation._parallel
            evaluation.run_evaluation(first)
            evaluation.run_evaluation(second)
            self.assertIs(evaluation._parallel, executor)
            pids = {ind.individual_fitness for ind in first + second}
            self.assertNotIn(main_pid, pids)
            self.assertLessEqual(len(pids), 4)

    def test_errors_propagate_and_context_closes(self):
        def objective(individual):
            raise ValueError('evaluation failed')

        for method in ('sequential', 'process'):
            evaluation = Evaluation(objective, method)
            with self.assertRaisesRegex(ValueError, 'evaluation failed'):
                with evaluation:
                    evaluation.run_evaluation([make_individual(1)])
            self.assertIsNone(evaluation._parallel)

    def test_configuration(self):
        for workers in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                Evaluation(lambda ind: 0, evaluation_workers=workers)
        with self.assertRaises(ValueError):
            Evaluation(lambda ind: 0, 'invalid')
        with self.assertRaises(ValueError):
            Evaluation(None)

    def test_optimizer_equivalence_and_verbose_statistics(self):
        results = []
        for method, verbose in (('sequential', False), ('process', False), ('sequential', True)):
            random.seed(42)
            np.random.seed(42)
            with contextlib.redirect_stdout(io.StringIO()):
                optimizer = GenethicOptimizer(
                    BOUNDS, 3, 20,
                    lambda ind: ind.get_individual_values()['x'] ** 2,
                    verbose=verbose, evaluation_method=method,
                )
            self.assertEqual(len(optimizer.POPULATION.generations_fitness_statistics_df), 3)
            results.append([
                [(ind.get_individual_values(), ind.individual_fitness) for ind in generation]
                for generation in optimizer.POPULATION.populuation_dict.values()
            ])
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])


if __name__ == '__main__':
    unittest.main()
