import pandas as pd
import plotly.graph_objects as go
from individuals import Individual
from typing import List, Dict, Tuple, Union, Literal
from info_tools import InfoTools
import scipy.stats as stats
import numpy as np
import time


class Population:
    def __init__(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], num_individuals: int):
        self.IT: InfoTools = InfoTools
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict
        self.num_individuals: int = num_individuals
        self.populuation_dict: Dict[str, List[Individual]] = {0: []}
        self.generations_fitness_statistics_df: pd.DataFrame | None = None

    def create_population(self) -> None:
        """
        Método para crear la población inicial
        :return:
        """
        while len(self.populuation_dict[0]) < self.num_individuals:
            individual: Individual = Individual(self.bounds_dict, None, 0)
            if not individual.malformation:
                self.populuation_dict[0].append(individual)

    def add_generation_population(self, children_list: List[Individual], generation: int) -> None:
        """
        Método para agregar una generación de individuos al diccionario de población
        :param children_list:
        :param generation:
        :return:
        """
        self.populuation_dict[generation] = children_list

    def get_generation_fitness_statistics(self, generation: int):
        """
            Calcula estadísticas descriptivas de una lista de valores numéricos.

            :param generation: Generación de la que vamos a obtener la info
            :return: Diccionario con media, mediana, desviación estándar, cuartiles, rango, moda y más.
            """
        values = np.array([z.individual_fitness for z in self.populuation_dict[generation] if not z.malformation])

        data = {
            "generation": [generation],
            "count": [len(values)],
            "mean": [np.mean(values)],
            "median": [np.median(values)],
            "std_dev": [np.std(values, ddof=1)],
            "variance": [np.var(values, ddof=1)],
            "min": [np.min(values)],
            "max": [np.max(values)],
            "range": [np.ptp(values)],
            "q1": [np.percentile(values, 25)],
            "q2": [np.percentile(values, 50)],  # Mediana
            "q3": [np.percentile(values, 75)],
            "iqr": [stats.iqr(values)],
            "mode": [stats.mode(values, keepdims=True)[0][0]]
        }

        if self.generations_fitness_statistics_df is None:
            self.generations_fitness_statistics_df = pd.DataFrame(data)
        else:
            self.generations_fitness_statistics_df = pd.concat([self.generations_fitness_statistics_df, pd.DataFrame(data)], axis=0)

        return self.generations_fitness_statistics_df

    def plot_generation_stats(self):
        fig = go.Figure()

        # Agregar líneas al gráfico
        show_stats = ['mean', 'median', 'mode', 'min', 'max', 'q1', 'q3']
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']

        for stat, color in zip(show_stats, colors):
            fig.add_trace(go.Scatter(
                x=self.generations_fitness_statistics_df['generation'],
                y=self.generations_fitness_statistics_df[stat],
                mode='lines+markers',
                name=stat.capitalize(),
                line=dict(color=color, width=2, shape='spline'),  # Líneas más suaves
                marker=dict(size=6, opacity=0.8)
            ))

        # Configuración del diseño
        fig.update_layout(
            title="Evolución de Estadísticas por Generación",
            xaxis=dict(title="Generación", showgrid=True, zeroline=False),
            yaxis=dict(title="Valor", showgrid=True, zeroline=False),
            template="plotly_white",  # Tema claro profesional
            legend=dict(title="Estadísticas", orientation="v", yanchor="top", xanchor="right"),
            margin=dict(l=50, r=200, t=50, b=50)  # Espacio extra a la derecha para la leyenda
        )

        fig.show()

    def animate_evolution_plotly(self, problem_type: Literal['minimize', 'maximize'] = "maximize", transition_duration_ms: int = 50) -> None:

        data_dict = self.populuation_dict
        generations = sorted(data_dict.keys())  # Obtener las generaciones en orden

        # Pongo el titulo dinamico
        title: str = f'Evolución de la Función de Fitness a lo largo de las generaciones. GEN: 0'

        # Determinar si es un problema de minimización o maximización
        is_minimization = True if problem_type == "minimize" else False

        # Preparar los datos para cada fotograma de la animación
        frames = []
        max_individuals = max(len(generation) for generation in data_dict.values())

        # Generar una paleta de colores única para cada individuo
        import colorsys
        def generate_distinct_colors(n):
            colors = []
            for i in range(n):
                hue = i / n
                saturation = 0.7
                lightness = 0.5
                rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
                colors.append(f'rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})')
            return colors

        # Preparar frames para la animación
        for generation in generations:
            # Extraer los valores de fitness de cada individuo en la generación
            population = data_dict[generation]
            fitness_values = [ind.individual_fitness for ind in population if ind.individual_fitness is not None]

            # Determinar el punto de inicio basado en minimización o maximización
            start_values = [1 if is_minimization else 0] * len(fitness_values)

            # Generar colores distintos para cada individuo
            colors = generate_distinct_colors(len(fitness_values))

            # Crear un frame que represente los valores de fitness para esta generación
            scatter_data = []

            # Scatter plot para individuos
            for i, (start, end, color) in enumerate(zip(start_values, fitness_values, colors)):
                scatter_data.append(go.Scatter(
                    x=[i],  # Posición del individuo
                    y=[start],  # Valor inicial
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=10,
                        opacity=0.8
                    ),
                    text=[f'Generation {generation}, Individual {i}<br>Final Fitness: {end}'],
                    hoverinfo='text',
                    name=f'Gen {generation} Ind {i}'
                ))

            # Frame para la animación de transición
            frame = go.Frame(
                data=scatter_data,
                name=f'Generation {generation}'
            )
            frames.append(frame)

        # Crear el layout inicial
        initial_data = []
        for i in range(max_individuals):
            initial_data.append(go.Scatter(
                x=[i],
                y=[1 if is_minimization else 0],
                mode='markers',
                marker=dict(color='gray', size=10, opacity=0.5)
            ))

        # Configurar la figura con animación
        fig = go.Figure(
            data=initial_data,
            layout=go.Layout(
                title=title,
                xaxis=dict(
                    title='Individuos',
                    range=[-1, max_individuals],
                    tickmode='linear',
                    tick0=0,
                    dtick=1
                ),
                yaxis=dict(
                    title='Fitness',
                    range=[-0.1, 1.1]  # Rango fijo como solicitaste
                ),
                updatemenus=[dict(
                    type='buttons',
                    x=0.1,
                    y=-0.2,
                    buttons=[dict(
                        label='Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': transition_duration_ms, 'redraw': True},  # 5 segundos por generación
                            'fromcurrent': True,
                            'transition': {
                                'duration': 5000,  # Duración de la transición
                                'easing': 'cubic-in-out'  # Tipo de animación suave
                            },
                            'mode': 'immediate'
                        }]
                    )]
                )],
                hovermode='closest',
            ),
            frames=frames
        )

        # Añadir fotogramas intermedios para la animación suave
        for generation in generations:
            population = data_dict[generation]
            fitness_values = [ind.individual_fitness for ind in population if ind.individual_fitness is not None]

            # Determinar el punto de inicio basado en minimización o maximización
            start_values = [1 if is_minimization else 0] * len(fitness_values)

            # Generar colores distintos para cada individuo
            colors = generate_distinct_colors(len(fitness_values))

            # Crear fotogramas de transición
            for progress in np.arange(0.0, 1.01, 0.01):
                transition_scatter_data = []
                for i, (start, end, color) in enumerate(zip(start_values, fitness_values, colors)):
                    # Interpolación lineal entre el inicio y el final
                    current_y = start + progress * (end - start)
                    transition_scatter_data.append(go.Scatter(
                        x=[i],
                        y=[current_y],
                        mode='markers',
                        marker=dict(
                            color=color,
                            size=10,
                            opacity=0.8
                        ),
                        text=[f'Generation: {generation}<br>Individual: {i}<br>Fitness: {end:.3f}<br>Progress: {progress * 100:.0f}%'],
                        hoverinfo='text',
                        name=f'Gen {generation} Ind {i}'
                    ))

                transition_frame = go.Frame(
                    data=transition_scatter_data,
                    name=f'Generation {generation} Progress {progress * 100:.0f}%'
                )
                frames.append(transition_frame)


        # Actualizar los frames de la figura
        fig.frames = frames

        # Mostrar la animación
        fig.show()














