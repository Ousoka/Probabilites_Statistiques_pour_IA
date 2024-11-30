import numpy as np
import matplotlib.pyplot as plt

# Données
data = np.array([
    [332, 428, 354, 1437, 526, 247, 427],
    [293, 559, 388, 1527, 567, 239, 258],
    [372, 767, 562, 1948, 927, 235, 433],
    [406, 563, 341, 1507, 544, 324, 407],
    [386, 608, 396, 1501, 568, 319, 363],
    [438, 843, 689, 2345, 1148, 243, 341],
    [534, 660, 367, 1620, 638, 414, 407],
    [460, 699, 484, 1856, 762, 400, 416],
    [385, 789, 621, 2366, 1149, 304, 282],
    [655, 776, 423, 1848, 759, 495, 486],
    [584, 995, 548, 2056, 893, 518, 319],
    [515, 1097, 887, 2630, 1167, 561, 284]
])

# Identifiants des catégories socio-professionnelles
labels = ["MA2", "EM2", "CA2", "MA3", "EM3", "CA3", "MA4", "EM4", "CA4", "MA5", "EM5", "CA5"]
variable_names = ["pain", "légumes", "fruits", "viande", "volaille", "lait", "vin"]

# Exécution de l'ACP en utilisant le code fourni
def calculate_center_of_gravity(X):
    return np.mean(X, axis=0)

def center_data(X, X_mean):
    return X - X_mean

def compute_covariance_matrix(X_centered):
    return np.dot(X_centered.T, X_centered) / (X_centered.shape[0])

def eigendecomposition(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

def project_data(X_centered, principal_components):
    return X_centered.dot(principal_components)

def calculate_contributions(principal_components, eigenvalues):
    n = principal_components.shape[0]
    contributions = np.zeros_like(principal_components)
    for j in range(principal_components.shape[1]):
        contributions[:, j] = (principal_components[:, j]**2) / ((n-1) * eigenvalues[j])
    return contributions * 100

def plot_principal_components(principal_components, explained_variance, labels):
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], color="blue", marker="o")
    for i, label in enumerate(labels):
        plt.annotate(label, (principal_components[i, 0], principal_components[i, 1]), textcoords="offset points", xytext=(5, 5), ha='center')
    plt.xlabel(f"Composante 1 ({explained_variance[0]:.2%} de la variance expliquée)")
    plt.ylabel(f"Composante 2 ({explained_variance[1]:.2%} de la variance expliquée)")
    plt.title("Projection des Individus - ACP")
    plt.grid(True)
    plt.show()

# def plot_correlation_circle(correlation_circle, explained_variance, variable_names):
#     fig, ax = plt.subplots(figsize=(8, 8))
#     circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', linewidth=0.5)
#     ax.add_artist(circle)
#     for i, (x, y) in enumerate(correlation_circle):
#         plt.arrow(0, 0, x, y, color='r', alpha=0.7, head_width=0.05, head_length=0.1)
#         plt.text(x, y, variable_names[i], color='g', ha='center', va='center', fontsize=12)
#     plt.xlim(-1.1, 1.1)
#     plt.ylim(-1.1, 1.1)
#     plt.axhline(0, color='grey', lw=0.5)
#     plt.axvline(0, color='grey', lw=0.5)
#     plt.xlabel(f"Composante 1 ({explained_variance[0]:.2%} de la variance expliquée)")
#     plt.ylabel(f"Composante 2 ({explained_variance[1]:.2%} de la variance expliquée)")
#     plt.title("Cercle des Corrélations - ACP")
#     plt.grid(False)
#     plt.show()

# def plot_correlation_circle(correlation_circle, explained_variance, variable_names):
#     """Trace le cercle de corrélations pour les variables sur les deux premières composantes principales."""
#     fig, ax = plt.subplots(figsize=(8, 8))
#     # Ajout du cercle unitaire
#     circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', linewidth=0.5)
#     ax.add_artist(circle)
    
#     # Tracé des vecteurs pour chaque variable
#     for i, (x, y) in enumerate(correlation_circle):
#         plt.arrow(0, 0, x, y, color='r', alpha=0.7, head_width=0.05, head_length=0.1)
#         plt.text(x * 1.1, y * 1.1, variable_names[i], color='g', ha='center', va='center', fontsize=12)

#     # Limites et styles du graphique
#     plt.xlim(-1.1, 1.1)
#     plt.ylim(-1.1, 1.1)
#     plt.axhline(0, color='grey', lw=0.5)
#     plt.axvline(0, color='grey', lw=0.5)
    
#     # Étiquettes des axes avec le pourcentage de variance expliquée
#     plt.xlabel(f"Composante 1 ({explained_variance[0]:.2%} de la variance expliquée)")
#     plt.ylabel(f"Composante 2 ({explained_variance[1]:.2%} de la variance expliquée)")
#     plt.title("Cercle des Corrélations - ACP")
    
#     # Ajustement de la grille pour améliorer la lisibilité
#     plt.grid(False)
#     plt.show()

# def plot_correlation_circle(correlation_circle, explained_variance, variable_names):
#     """Trace le cercle de corrélations pour les variables sur les deux premières composantes principales."""
#     fig, ax = plt.subplots(figsize=(8, 8))
#     # Ajout du cercle unitaire pour visualiser les limites
#     circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', linewidth=0.5)
#     ax.add_artist(circle)
    
#     # Tracé des flèches représentant les corrélations pour chaque variable
#     for i, (x, y) in enumerate(correlation_circle):
#         # Flèche depuis l'origine (0,0) jusqu'à (x,y)
#         plt.arrow(0, 0, x, y, color='r', alpha=0.7, head_width=0.05, head_length=0.1)
#         # Positionnement du nom de la variable à l'extrémité de chaque flèche
#         plt.text(x * 1.05, y * 1.05, variable_names[i], color='g', ha='center', va='center', fontsize=12)

#     # Limites et styles de la figure pour cadrer le cercle
#     plt.xlim(-1.1, 1.1)
#     plt.ylim(-1.1, 1.1)
#     plt.axhline(0, color='grey', lw=0.5)
#     plt.axvline(0, color='grey', lw=0.5)
    
#     # Étiquettes des axes avec le pourcentage de variance expliquée
#     plt.xlabel(f"Composante 1 ({explained_variance[0]:.2%} de la variance expliquée)")
#     plt.ylabel(f"Composante 2 ({explained_variance[1]:.2%} de la variance expliquée)")
#     plt.title("Cercle des Corrélations - ACP")
    
#     # Ajustement de la grille pour une meilleure lisibilité
#     plt.grid(False)
#     plt.show()

# Normalisation des coordonnées pour le cercle des corrélations
def plot_correlation_circle(correlation_circle, explained_variance, variable_names):
    """Trace le cercle de corrélations pour les variables sur les deux premières composantes principales."""
    
    # Normalisation des vecteurs des variables pour le cercle
    norms = np.linalg.norm(correlation_circle, axis=0)
    correlation_circle_normalized = correlation_circle / norms

    # Création du graphique
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0, 0), 1, color='b', fill=False, linestyle='--', linewidth=0.5)
    ax.add_artist(circle)

    # Plot les vecteurs pour chaque variable
    for i in range(correlation_circle_normalized.shape[1]):
        # Dessine l'arrow
        plt.arrow(0, 0, correlation_circle_normalized[0, i], correlation_circle_normalized[1, i], 
                  color='r', alpha=0.7, head_width=0.05, head_length=0.1)
        
        # Positionne les labels sur chaque vecteur, à 70% de la longueur de l'arrow
        label_x = 0.7 * correlation_circle_normalized[0, i]
        label_y = 0.7 * correlation_circle_normalized[1, i]
        
        # Ajoute le label à l'emplacement calculé
        plt.text(label_x, label_y, variable_names[i], color='g', ha='center', va='center', fontsize=12,
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # Configuration du graphique
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    
    # Titre et labels des axes
    plt.xlabel(f"Composante 1 ({explained_variance[0]:.2%} de la variance expliquée)")
    plt.ylabel(f"Composante 2 ({explained_variance[1]:.2%} de la variance expliquée)")
    plt.title("Cercle des Corrélations - ACP")
    
    # Affichage
    plt.grid(False)
    plt.show()

# # Exemple de données projetées après l'ACP (à remplacer par les résultats réels)
# # correlation_circle est un tableau (2, n) où n est le nombre de variables
# correlation_circle = np.array([
#     [0.5, 0.6, 0.7, 0.3, 0.4, 0.2, 0.6],  # Composante 1 (ajoutez ici la 7ème variable si nécessaire)
#     [0.4, 0.7, 0.5, 0.6, 0.3, 0.5, 0.4]   # Composante 2 (ajoutez ici la 7ème variable si nécessaire)
# ])

# # Exemple de noms des variables (ajoutez ici tous les noms de variables)
# variable_names = ["pain", "légumes", "fruits", "viande", "volaille", "lait", "vin"]

# # Variance expliquée (à remplacer par les résultats réels)
# explained_variance = [0.45, 0.25]

# # Appel de la fonction pour afficher le cercle des corrélations
# plot_correlation_circle(correlation_circle, explained_variance, variable_names)


def pca(X, num_components=2):
    X_mean = calculate_center_of_gravity(X)
    X_centered = center_data(X, X_mean)
    cov_matrix = compute_covariance_matrix(X_centered)
    eigenvalues, eigenvectors = eigendecomposition(cov_matrix)
    principal_components = eigenvectors[:, :num_components]
    X_transformed = project_data(X_centered, principal_components)
    explained_variance = eigenvalues / np.sum(eigenvalues)
    return X_transformed, explained_variance, eigenvalues, eigenvectors, X_centered

# Calcul de l'ACP
num_components = 2
principal_components, explained_variance, eigenvalues, eigenvectors, X_centered = pca(data, num_components)

# Affichage des contributions des individus
contributions = calculate_contributions(principal_components, eigenvalues)
print("\nContributions des individus (en %):")
for i, (c1, c2) in enumerate(contributions):
    print(f"{labels[i]} - Axe 1: {c1:.2f}%, Axe 2: {c2:.2f}%")

# Affichage graphique des individus dans le nouvel espace
plot_principal_components(principal_components, explained_variance, labels)

# Calcul du cercle des corrélations
correlation_circle = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])

# Affichage du cercle des corrélations
plot_correlation_circle(correlation_circle, explained_variance, variable_names)
