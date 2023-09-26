import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.cluster import contingency_matrix, silhouette_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment as linear_assignment

from sklearn.metrics import accuracy_score, rand_score, adjusted_rand_score

# Dados
from sklearn.datasets import load_iris, load_wine

def get_iris_data():
    X = load_iris().data
    y = load_iris().target
    target_names = load_iris().target_names

    return (X, y, target_names)

def get_wine_data():
    X = load_wine().data
    y = load_wine().target
    target_names = load_wine().target_names

    return (X, y,  target_names)

def get_synthetic_control_data():
    path = 'synthetic_control.data'

    X = []

    with open(path, 'r') as file:
        for line in file.readlines():
            X.append([float(value) for value in line.split()])

    X = np.array(X)
    y = np.concatenate( [ [i] * 100 for i in range(6) ])
    target_names = ['Normal','Cyclic','Increasing trend','Decreasing trend','Upward shift','Downward shift']

    return(X, y, target_names)

def plot_cluster_evaluation(X, labels_true, labels_pred, target_names):
    classes = np.unique(labels_true)

    # Criar um scatter plot
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=labels_pred, cmap='viridis')
    plt.title("Plot em duas dimensões da clusterização")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    # Criar uma matriz de contingência
    plt.subplot(1, 3, 2)
    matriz_contingencia = contingency_matrix(labels_true, labels_pred)
    sns.heatmap(matriz_contingencia, annot=True, fmt='d', cmap='YlGnBu', cbar=False, xticklabels=np.unique(labels_pred), yticklabels=target_names)
    plt.title("Matriz de Contingência")
    plt.xlabel("Clusters (Previstos)")
    plt.ylabel("Classes (Reais)")

    row_ind, col_ind = linear_assignment(matriz_contingencia, maximize = True)

    #print(f'row_ind {row_ind}')
    #print(f'col_ind {col_ind}')

    sum = 0
    # Somando o maior valor das colunas (A posição do maior valor está na variável col_ind)
    for i, j in zip(row_ind, col_ind):
        sum += matriz_contingencia[i][j]

    accuracy = (sum / len(labels_true))*100

     # Calcular e exibir as métricas de avaliação de clusters
    plt.subplot(1, 3, 3)
    silhouette = silhouette_score(X, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    rand_index = rand_score(labels_true, labels_pred)
    adj_rand_score = adjusted_rand_score(labels_true, labels_pred)

    plt.text(0.5, 0.5, f"Silhouette Score: {silhouette:.2f}\n Normalized Mutual Information: {nmi:.2f}\n Accuracy Score: {accuracy:.2f}%\n Rand Index: {rand_index:.2f}\n Adjusted Rand Score: {adj_rand_score:.2f}",
             fontsize=12, ha='center')
    plt.axis('off')
    plt.title("Métricas de Avaliação de Clusters")

    plt.tight_layout()
    plt.show()


