import numpy as np

# Matrices de données A et B
A = np.array([
    [2, -3, -5],
    [-2, 2, 0],
    [0, 4, 2],
    [4, 0, 2],
    [2, -2, 0],
    [-6, -1, 1]
])

B = np.array([
    [20, 15],  # I1
    [5, 2],    # I2
    [12, 21],  # I3
    [21, 13],  # I4
    [2, 7],    # I5
    [12, 20]   # I6
])

# Fonction pour centrer les matrices
def center_matrix(X):
    mean_X = np.mean(X, axis=0)
    centered_X = X - mean_X
    return centered_X

# Centrage des matrices
A_centered = center_matrix(A)
B_centered = center_matrix(B)

# Calcul de la matrice de variance-covariance pour A et B
cov_matrix_A = np.dot(A_centered.T, A_centered) / A_centered.shape[0]
cov_matrix_B = np.dot(B_centered.T, B_centered) / B_centered.shape[0]

# Étude spectrale : valeurs propres et vecteurs propres
eigenvalues_A, eigenvectors_A = np.linalg.eig(cov_matrix_A)
eigenvalues_B, eigenvectors_B = np.linalg.eig(cov_matrix_B)

# Normalisation des vecteurs propres pour s'assurer qu'ils sont de norme 1
eigenvectors_A = eigenvectors_A / np.linalg.norm(eigenvectors_A, axis=0)
eigenvectors_B = eigenvectors_B / np.linalg.norm(eigenvectors_B, axis=0)

# Projection des données sur les nouvelles composantes principales
A_projected = np.dot(A_centered, eigenvectors_A)
B_projected = np.dot(B_centered, eigenvectors_B)

# Résultats
print("Matrice centrée A :\n", A_centered)
print("\nMatrice de variance-covariance A :\n", cov_matrix_A)
print("\nValeurs propres de A :\n", eigenvalues_A)
print("\nVecteurs propres normalisés de A :\n", eigenvectors_A)
print("\nProjection de A sur la carte factorielle :\n", A_projected)

print("\nMatrice centrée B :\n", B_centered)
print("\nMatrice de variance-covariance B :\n", cov_matrix_B)
print("\nValeurs propres de B :\n", eigenvalues_B)
print("\nVecteurs propres normalisés de B :\n", eigenvectors_B)
print("\nProjection de B sur la carte factorielle :\n", B_projected)
