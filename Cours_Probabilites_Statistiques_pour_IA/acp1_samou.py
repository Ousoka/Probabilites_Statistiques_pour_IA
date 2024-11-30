import numpy as np

def input_matrix():
    """Demande à l'utilisateur d'entrer la matrice de données."""
    while True:
        try:
            n = int(input("Entrez le nombre de lignes (observations) : "))
            p = int(input("Entrez le nombre de colonnes (variables) : "))
            break
        except ValueError:
            print("Veuillez entrer des entiers valides pour les dimensions.")
    
    data = []
    print("Entrez les éléments de la matrice, ligne par ligne :")
    for i in range(n):
        row = []
        for j in range(p):
            while True:
                try:
                    value = float(input(f"Élément ({i+1},{j+1}) : "))
                    row.append(value)
                    break
                except ValueError:
                    print("Veuillez entrer un nombre valide.")
        data.append(row)
    
    return np.array(data)

def calculate_center_of_gravity(X):
    """Calcule et retourne le centre de gravité du jeu de données."""
    return np.mean(X, axis=0)

def center_data(X, X_mean):
    """Centre les données en soustrayant le centre de gravité."""
    return X - X_mean

def compute_covariance_matrix(X_centered):
    """Calcule la matrice de covariance des données centrées."""
    return np.cov(X_centered, rowvar=False)

def eigendecomposition(cov_matrix):
    """Calcule les valeurs et vecteurs propres, et les trie par valeur propre décroissante."""
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

def project_data(X_centered, principal_components):
    """Projette les données dans l'espace des composantes principales."""
    return X_centered.dot(principal_components)

def pca(X, num_components):
    """Effectue l'ACP complète et retourne les résultats intermédiaires et finaux."""
    # Calcul du centre de gravité
    X_mean = calculate_center_of_gravity(X)
    
    # Centrage des données
    X_centered = center_data(X, X_mean)
    
    # Calcul de la matrice de covariance
    cov_matrix = compute_covariance_matrix(X_centered)
    
    # Décomposition en valeurs et vecteurs propres
    eigenvalues, eigenvectors = eigendecomposition(cov_matrix)
    
    # Sélection des composantes principales
    principal_components = eigenvectors[:, :num_components]
    
    # Projection des données
    X_transformed = project_data(X_centered, principal_components)
    
    # Affichage des résultats
    print("\n--- Résultats de l'ACP ---")
    print("Centre de gravité (g) :\n", X_mean)
    print("\nDonnées centrées (X - g) :\n", X_centered)
    print("\nMatrice de covariance :\n", cov_matrix)
    print("\nValeurs propres triées :\n", eigenvalues)
    print("\nVecteurs propres triés :\n", eigenvectors)
    print(f"\nLes {num_components} composantes principales :\n", principal_components)
    print("\nDonnées projetées dans le nouvel espace :\n", X_transformed)
    
    # Retourner les valeurs pour usage ultérieur si nécessaire
    return {
        'center_of_gravity': X_mean,
        'centered_data': X_centered,
        'covariance_matrix': cov_matrix,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'principal_components': principal_components,
        'transformed_data': X_transformed
    }

# Utilisation
# X = input_matrix()

X = np.array([
    [8, 1, 0],
    [4, 6, 5],
    [6, 8, 7],
    [10, 4, 7],
    [8, 2, 5],
    [0, 3, 6]
])

# Demander le nombre de composantes principales à conserver
while True:
    try:
        num_components = int(input("Entrez le nombre de composantes principales à garder (k) : "))
        if 1 <= num_components <= X.shape[1]:
            break
        else:
            print("Le nombre de composantes principales doit être entre 1 et le nombre de variables.")
    except ValueError:
        print("Veuillez entrer un entier valide pour le nombre de composantes principales.")

# Exécuter l'ACP et obtenir les résultats
results = pca(X, num_components)
