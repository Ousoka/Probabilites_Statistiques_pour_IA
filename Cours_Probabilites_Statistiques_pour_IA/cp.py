import numpy as np

# Définir les matrices de données A et B
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

# Centrer les matrices A et B
A_centered = center_matrix(A)
B_centered = center_matrix(B)

# Calculer la matrice de covariance
cov_matrix_A = np.cov(A_centered, rowvar=False)
cov_matrix_B = np.cov(B_centered, rowvar=False)

# Calculer les valeurs propres et vecteurs propres
eigenvalues_A, eigenvectors_A = np.linalg.eig(cov_matrix_A)
eigenvalues_B, eigenvectors_B = np.linalg.eig(cov_matrix_B)

# Calculer les composantes principales (projections sur les axes principaux)
A_principal_components = np.dot(A_centered, eigenvectors_A)
B_principal_components = np.dot(B_centered, eigenvectors_B)

# Résultats
print("Composantes principales de A :\n", A_principal_components)
print("\nComposantes principales de B :\n", B_principal_components)
