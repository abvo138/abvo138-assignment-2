import numpy as np
import dash
import pandas as pd
from dash import dcc, html, Input, Output, State
import plotly.express as px

class KMeans:
    def __init__(self, data, k, option):
        self.data = data
        self.k = k
        self.option = option
        self.assignment = [-1 for _ in range(len(data))]
        self.centers = None
        self.figures = []

    def snap(self):
        fig = px.scatter(x=self.data[:, 0], y=self.data[:, 1], color=self.assignment, 
                         title="KMeans Clustering", labels={'x': 'X-axis', 'y': 'Y-axis'}, 
                         color_continuous_scale='Viridis')
        if self.centers is not None:
            fig.add_scatter(x=self.centers[:, 0], y=self.centers[:, 1], mode='markers', 
                            marker=dict(color='red', size=10), name='Centroids')
        fig.update_layout(coloraxis_showscale=False)
        self.figures.append(fig)

    def initialize(self, manual_centers=None):
        if self.option == "manual" and manual_centers is not None:
            self.centers = np.array(manual_centers)
        
        elif self.option == "random":
            indices = np.random.choice(len(self.data), size=self.k, replace=False)
            self.centers = self.data[indices]

        elif self.option == "farthest first":
            centroids = []
            initial_centroid_choice = np.random.choice(len(self.data), size=1)
            centroids.append(self.data[initial_centroid_choice[0]].copy())
            for _ in range(1, self.k):
                distances = np.min([np.linalg.norm(self.data - centroid, axis=1) for centroid in centroids], axis=0)
                furthest_point_index = np.argmax(distances)
                centroids.append(self.data[furthest_point_index].copy())
            self.centers = np.array(centroids)

        elif self.option == "Kmeans++":
            centroids = []
            initial_centroid_choice = np.random.choice(len(self.data), size=1)
            centroids.append(self.data[initial_centroid_choice[0]].copy())

            for _ in range(1, self.k):
                distance = [np.min([np.linalg.norm(point - c) for c in centroids]) ** 2 for point in self.data]
                total = sum(distance)
                probabilities = [d / total for d in distance]
                new_centroid_index = np.random.choice(range(len(self.data)), p=probabilities)
                centroids.append(self.data[new_centroid_index].copy())

            self.centers = np.array(centroids)

        self.snap()

    def make_clusters(self):
        for i in range(len(self.data)):
            self.assignment[i] = np.argmin([np.linalg.norm(self.data[i] - center) for center in self.centers])

    def compute_centers(self):
        new_centers = []
        for i in range(self.k):
            cluster = [self.data[j] for j in range(len(self.assignment)) if self.assignment[j] == i]
            if cluster:
                new_centers.append(np.mean(cluster, axis=0))
            else:
                new_centers.append(self.centers[i])
        return np.array(new_centers)
    
    def lloyd_step(self):
        if self.centers is None:
            self.initialize()
            return False

        self.make_clusters()
        new_centers = self.compute_centers()
        self.snap()
        
        if np.allclose(self.centers, new_centers):
            return True
        self.centers = new_centers
        return False

    def converge(self):
        converged = False
        while not converged:
            converged = self.lloyd_step()
        return self.figures[-1]

# Generate synthetic data manually
def generate_blobs(n_samples=100, n_clusters=3, cluster_std=1.0, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    data = []
    centers = []
    for _ in range(n_clusters):
        center = np.random.uniform(-10, 10, size=(2,))  # Randomly choosing center in range [-10, 10]
        cluster_points = center + np.random.randn(n_samples // n_clusters, 2) * cluster_std
        data.append(cluster_points)
        centers.append(center)

    data = np.vstack(data)
    return data

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("KMeans Clustering Step-Through"),
    dcc.Input(id='k-input', type='number', value=4, min=1, max=10),
    dcc.Dropdown(
        id='option-dropdown',
        options=[
            {'label': 'Random', 'value': 'random'},
            {'label': 'Farthest First', 'value': 'farthest first'},
            {'label': 'Kmeans++', 'value': 'Kmeans++'},
            {'label': 'Manual', 'value': 'manual'}, 
        ],
        value='farthest first'
    ),
    html.Button('Generate Data', id='generate-data-button'),
    html.Button('Animate Steps', id='animate-button', disabled=True),
    html.Button('Converge', id='converge-button', disabled=True),
    html.Button('Reset', id='reset-button'),
    dcc.Graph(id='kmeans-graph'),
    dcc.Interval(id='step-interval', interval=1000, n_intervals=0, disabled=True)
])

# Global variables
dataset = None
kmeans_instance = None
manual_centers = []

@app.callback(
    [Output('kmeans-graph', 'figure'),
     Output('converge-button', 'disabled'),
     Output('animate-button', 'disabled'),
     Output('step-interval', 'disabled')],
    [Input('generate-data-button', 'n_clicks'),
     Input('kmeans-graph', 'clickData'),
     Input('converge-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('step-interval', 'n_intervals'),
     Input('animate-button', 'n_clicks')],
    [State('k-input', 'value'),
     State('option-dropdown', 'value'),
     State('step-interval', 'disabled')]
)
def handle_graph_update(generate_clicks, clickData, converge_clicks, reset_clicks, n_intervals, animate_clicks, k, option, interval_disabled):
    global dataset, kmeans_instance, manual_centers

    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'generate-data-button':
        dataset = generate_blobs(n_samples=100, n_clusters=k, cluster_std=1.0)
        manual_centers = []

        kmeans_instance = KMeans(dataset, k, option)

        if option != 'manual':
            kmeans_instance.initialize()
            return kmeans_instance.figures[-1], False, False, True

        return px.scatter(x=dataset[:, 0], y=dataset[:, 1]), True, True, True

    elif kmeans_instance is None:
        return px.scatter(), True, True, True

    elif option == 'manual' and triggered_id == 'kmeans-graph' and clickData is not None:
        clicked_point = clickData['points'][0]
        x, y = clicked_point['x'], clicked_point['y']

        if len(manual_centers) < k:
            manual_centers.append([x, y])

        kmeans_instance.centers = np.array(manual_centers)
        kmeans_instance.snap()

        if len(manual_centers) == k:
            return kmeans_instance.figures[-1], False, False, True

        return kmeans_instance.figures[-1], True, True, True

    elif triggered_id == 'converge-button':
        final_figure = kmeans_instance.converge()
        return final_figure, True, True, True

    elif triggered_id == 'animate-button':
        return dash.no_update, True, True, not interval_disabled

    elif triggered_id == 'step-interval' and not interval_disabled:
        converged = kmeans_instance.lloyd_step()
        if converged:
            return kmeans_instance.figures[-1], True, True, True
        return kmeans_instance.figures[-1], True, True, False

    elif triggered_id == 'reset-button':
        manual_centers = []
        kmeans_instance = KMeans(dataset, k, option)
        if option != 'manual':
            kmeans_instance.initialize()
            return kmeans_instance.figures[-1], False, False, True
        return px.scatter(x=dataset[:, 0], y=dataset[:, 1]), True, True, True

    return kmeans_instance.figures[-1], True, True, True


if __name__ == '__main__':
    app.run_server(debug=True, port=3000)
