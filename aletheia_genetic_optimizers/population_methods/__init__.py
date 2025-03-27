import colorsys
import pandas as pd
import plotly.graph_objects as go
from aletheia_genetic_optimizers.individuals import Individual
from typing import List, Dict, Tuple, Union, Literal
from info_tools import InfoTools
import scipy.stats as stats
import numpy as np

# TODO: Crear un metodo para ir viendo la distribucion de los parametros a optimizar
class Population:
    def __init__(self, bounds_dict: Dict[str, Tuple[Union[int, float]]], num_individuals: int, problem_restrictions: Literal['bound_restricted', 'full_restricted']):
        self.IT: InfoTools = InfoTools
        self.bounds_dict: Dict[str, Tuple[Union[int, float]]] = bounds_dict
        self.num_individuals: int = num_individuals
        self.problem_restrictions: Literal['bound_restricted', 'full_restricted'] = problem_restrictions
        self.populuation_dict: Dict[str, List[Individual]] = {0: []}
        self.generations_fitness_statistics_df: pd.DataFrame | None = None

    def create_population(self) -> None:
        """
        Método para crear la población inicial
        :return:
        """

        while len(self.populuation_dict[0]) < self.num_individuals:
            individual: Individual = Individual(self.bounds_dict, None, 0, self.problem_restrictions)
            if not individual.malformation:
                self.populuation_dict[0].append(individual)

    # <editor-fold desc="Getters y setters    ----------------------------------------------------------------------------------------------------------------------------------">

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

    # <editor-fold desc="Metodos de graficación    -------------------------------------------------------------------------------------------------------------------------------">
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
            title=dict(
                text="Evolución de Estadísticas por Generación",
                font=dict(
                    size=22,
                    family='Arial, sans-serif',
                    color='#2C3E50'  # Azul profundo elegante
                ),
                x=0.5,
                xanchor='center',
                pad=dict(b=20, t=20)
            ),
            plot_bgcolor='rgba(245,248,250,0.9)',  # Gris azulado muy claro
            paper_bgcolor='white',
            xaxis=dict(
                title="Generación",
                showgrid=True,
                zeroline=False,
                gridcolor='lightgrey',
                zerolinecolor='lightgrey',
                showline=True,
                linewidth=2,
                linecolor='#BDC3C7'
            ),
            yaxis=dict(
                title="Valor",
                showgrid=True,
                zeroline=False,
                gridcolor='lightgrey',
                zerolinecolor='lightgrey',
                showline=False,
                linewidth=2,
                linecolor='#BDC3C7'
            ),
            legend=dict(
                title='Estadísticas',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#BDC3C7',
                borderwidth=1,
                x=1.05,  # Mover leyenda fuera del gráfico
                xanchor='left'
            ),
            margin=dict(l=50, r=200, t=50, b=50)  # Espacio extra a la derecha para la leyenda
        )

        fig.show()

    def plot_evolution_animated(self, problem_type: Literal['minimize', 'maximize'] = "maximize", transition_duration_ms: int = 50) -> None:

        data_dict = self.populuation_dict
        generations = sorted(data_dict.keys())  # Obtener las generaciones en orden

        # Determinar si es un problema de minimización o maximización
        is_minimization = True if problem_type == "minimize" else False

        # Obtengo losfitness mas grandes y mas pequeños
        max_fitness_value: float = max(
            [ind.individual_fitness for ind in [elemento for sublista in data_dict.values() for elemento in sublista] if ind.individual_fitness is not None])
        min_fitness_value: float = min(
            [ind.individual_fitness for ind in [elemento for sublista in data_dict.values() for elemento in sublista] if ind.individual_fitness is not None])

        # Preparar los datos para cada fotograma de la animación
        frames = []
        max_individuals = max(len(generation) for generation in data_dict.values())

        def generate_distinct_colors(n):
            colors = []
            for i in range(n):
                hue = i / n
                saturation = 0.7
                lightness = 0.5
                rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
                colors.append(f'rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})')
            return colors

        # Generar un color por generación
        generation_colors = generate_distinct_colors(len(generations))
        generation_color_map = dict(zip(generations, generation_colors))

        # Crear el layout inicial con individuos de color de su generación
        initial_data = []
        for generation in generations:
            population = data_dict[generation]
            gen_color = generation_color_map[generation]
            for i, ind in enumerate(population):
                initial_data.append(go.Scatter(
                    x=[i],
                    y=[ind.individual_fitness],  # Usar el fitness real del individuo
                    mode='markers',
                    marker=dict(
                        color=gen_color,
                        size=10,
                        opacity=0.7,
                        line=dict(width=1.5, color='DarkSlateGrey')
                    ),
                    name=f'Gen {generation} Ind {i}',
                    text=[f'Gen: {generation}<br>Ind: {i}<br>Fitness: {ind.individual_fitness if ind.individual_fitness is not None else 0:.3f}'],
                    hoverinfo='text',
                    textposition='top center'
                ))

        # Preparar frames para la animación
        for generation in generations:
            # Extraer los valores de fitness de cada individuo en la generación
            population = data_dict[generation]
            fitness_values = [ind.individual_fitness for ind in population if ind.individual_fitness is not None]

            # Determinar el punto de inicio basado en minimización o maximización
            start_values = [1 if is_minimization else 0] * len(fitness_values)

            # Color de la generación actual
            gen_color = generation_color_map[generation]

            # Crear un frame que represente los valores de fitness para esta generación
            scatter_data = []

            # Scatter plot para individuos
            for i, (start, end, color) in enumerate(zip(start_values, fitness_values, [gen_color] * len(fitness_values))):
                scatter_data.append(go.Scatter(
                    x=[i],  # Posición del individuo
                    y=[start],  # Valor inicial
                    mode='markers+text',
                    marker=dict(
                        color=color,
                        size=10,
                        opacity=0.8
                    ),
                    text=[f'Gen: {generation}<br>Ind: {i}<br>Fitness: {end:.3f}'],
                    textposition='top center',
                    hoverinfo='text',
                    hoverlabel=dict(bgcolor='white', font_size=12),
                    name=f'Gen {generation} Ind {i}'
                ))

            # Frame para la animación de transición
            frame = go.Frame(
                data=scatter_data,
                name=f'Generation {generation}'
            )
            frames.append(frame)

        # Configurar la figura con animación
        fig = go.Figure(
            data=initial_data,
            layout=go.Layout(
                # Título con estilo profesional
                title=dict(
                    text='Evolución de la Función de Fitness a lo largo de las generaciones',
                    font=dict(
                        size=22,
                        family='Arial, sans-serif',
                        color='#2C3E50'  # Azul profundo elegante
                    ),
                    x=0.5,
                    xanchor='center',
                    pad=dict(b=20, t=20)
                ),

                # Fondo profesional
                plot_bgcolor='rgba(245,248,250,0.9)',  # Gris azulado muy claro
                paper_bgcolor='white',

                # Menú de animación centrado y con estilo
                updatemenus=[dict(
                    type='buttons',
                    x=0.5,  # Centrado horizontalmente
                    xanchor='center',
                    y=0.95,  # Posicionado justo debajo del título
                    buttons=[dict(
                        label='▶ Iniciar Animación',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': transition_duration_ms, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {
                                'duration': 5000,
                                'easing': 'cubic-in-out'
                            },
                            'mode': 'immediate'
                        }]
                    )]
                )],

                xaxis=dict(
                    title='Individuos',
                    range=[-1, max_individuals],
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    gridcolor='lightgrey',
                    zerolinecolor='lightgrey',
                    showline=True,
                    linewidth=2,
                    linecolor='#BDC3C7'
                ),
                yaxis=dict(
                    title='Fitness',
                    range=[-0.1 if min_fitness_value > -0.1 else min_fitness_value, 1.1 if max_fitness_value < 1.1 else max_fitness_value],
                    gridcolor='lightgrey',
                    zerolinecolor='lightgrey',
                    showline=False,
                    linewidth=2,
                    linecolor='#BDC3C7'
                ),

                # Leyenda con estilo
                legend=None,
                showlegend=False,

                hovermode='closest',
            ),
            frames=frames
        )

        # Añadir fotogramas intermedios para la animación suave
        for generation in generations:
            population = data_dict[generation]
            fitness_values = [ind.individual_fitness for ind in population if ind.individual_fitness is not None]

            # Determinar el punto de inicio basado en minimización o maximización
            start_values = [(1 if max_fitness_value < 1 else max_fitness_value) if is_minimization else (0 if min_fitness_value > 0 else min_fitness_value)] * len(fitness_values)

            # Color de la generación actual
            gen_color = generation_color_map[generation]

            # Crear fotogramas de transición
            start_progress_limit: float = (1 if max_fitness_value < 1 else max_fitness_value) if is_minimization else (0 if min_fitness_value > 0 else min_fitness_value)
            end_progress_limit: float = (1 if max_fitness_value < 1 else max_fitness_value) if not is_minimization else (0 if min_fitness_value > 0 else min_fitness_value)
            animation_range: float = (end_progress_limit - start_progress_limit) / 100
            for progress in np.arange(start_progress_limit, end_progress_limit, animation_range):
                transition_scatter_data = []
                for i, (start, end, color) in enumerate(zip(start_values, fitness_values, [gen_color] * len(fitness_values))):
                    # Interpolación lineal entre el inicio y el final
                    current_y = start + progress * (end - start)
                    transition_scatter_data.append(go.Scatter(
                        x=[i],
                        y=[current_y],
                        mode='markers+text',
                        marker=dict(
                            color=color,
                            size=12,
                            opacity=0.8
                        ),
                        text=[f'<b>[Gen]</b>:<br>{generation}<br><b>[Ind]</b>:<br>{i}<br><b>[Fitness]</b><br>{end:.3f}'],
                        textposition='top center',
                        hoverinfo='text',
                        hoverlabel=dict(bgcolor='white', font_size=10),
                        name=f'Gen {generation} Ind {i}'
                    ))

                transition_frame = go.Frame(
                    data=transition_scatter_data,
                    name=f'Generation {generation} Progress {progress * 100:.0f}%'
                )

                if progress == start_progress_limit or progress == end_progress_limit:
                    for i in range(15):
                        frames.append(transition_frame)
                frames.append(transition_frame)

        # Actualizar los frames de la figura
        fig.frames = frames

        # Mostrar la animación
        fig.show()

    def plot_evolution(self) -> None:

        data_dict = self.populuation_dict
        generations = sorted(data_dict.keys())  # Obtener las generaciones en orden

        # Preparar los datos para cada fotograma de la animación
        max_individuals = max(len(generation) for generation in data_dict.values())

        # Obtengo losfitness mas grandes y mas pequeños
        max_fitness_value: float = max(
            [ind.individual_fitness for ind in [elemento for sublista in data_dict.values() for elemento in sublista] if ind.individual_fitness is not None])
        min_fitness_value: float = min(
            [ind.individual_fitness for ind in [elemento for sublista in data_dict.values() for elemento in sublista] if ind.individual_fitness is not None])

        def generate_distinct_colors(n):
            colors = []
            for i in range(n):
                hue = i / n
                saturation = 0.7
                lightness = 0.5
                rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
                colors.append(f'rgb({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)})')
            return colors

        # Generar un color por generación
        generation_colors = generate_distinct_colors(len(generations))
        generation_color_map = dict(zip(generations, generation_colors))

        # Crear el layout inicial con individuos de color de su generación
        initial_data = []
        for generation in generations:
            population = data_dict[generation]
            gen_color = generation_color_map[generation]
            for i, ind in enumerate(population):
                initial_data.append(go.Scatter(
                    x=[i],
                    y=[ind.individual_fitness],  # Usar el fitness real del individuo
                    mode='markers',
                    marker=dict(
                        color=gen_color,
                        size=14,
                        opacity=0.7,
                        line=dict(width=1.5, color='DarkSlateGrey')
                    ),
                    name=f'Gen {generation} Ind {i}',
                    text=[f'Gen: {generation}<br>Ind: {i}<br>Fitness: {ind.individual_fitness if ind.individual_fitness is not None else 0:.3f}'],
                    hoverinfo='text',
                    textposition='top center'
                ))

        # Configurar la figura con animación
        fig = go.Figure(
            data=initial_data,
            layout=go.Layout(
                # Título con estilo profesional
                title=dict(
                    text='Evolución de la Función de Fitness a lo largo de las generaciones',
                    font=dict(
                        size=22,
                        family='Arial, sans-serif',
                        color='#2C3E50'  # Azul profundo elegante
                    ),
                    x=0.5,
                    xanchor='center',
                    pad=dict(b=20, t=20)
                ),

                # Fondo profesional
                plot_bgcolor='rgba(245,248,250,0.9)',  # Gris azulado muy claro
                paper_bgcolor='white',

                xaxis=dict(
                    title='Individuos',
                    range=[-1, max_individuals],
                    tickmode='linear',
                    tick0=0,
                    dtick=1,
                    gridcolor='lightgrey',
                    zerolinecolor='lightgrey',
                    showline=True,
                    linewidth=2,
                    linecolor='#BDC3C7'
                ),
                yaxis=dict(
                    title='Fitness',
                    range=[-0.1 if min_fitness_value > -0.1 else min_fitness_value, 1.1 if max_fitness_value < 1.1 else max_fitness_value],
                    gridcolor='lightgrey',
                    zerolinecolor='lightgrey',
                    showline=False,
                    linewidth=2,
                    linecolor='#BDC3C7'
                ),

                # Leyenda con estilo
                legend=None,
                showlegend=False,
                hovermode='closest',
            ),
        )

        # Mostrar la animación
        fig.show()

    # </editor-fold>
