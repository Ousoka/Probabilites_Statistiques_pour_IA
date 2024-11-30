import numpy as np

# Matrices des données
V = np.array([
    [2, -3, -5],
    [-2, 2, 0],
    [0, 4, 2],
    [4, 0, 2],
    [2, -2, 0],
    [-6, -1, 1]
])

W = np.array([
    [20, 15],  # I1
    [5, 2],    # I2
    [12, 21],  # I3
    [21, 13],  # I4
    [2, 7],    # I5
    [12, 20]   # I6
])

# 1. Centrer les données
V_centree = V - V.mean(axis=0)
W_centree = W - W.mean(axis=0)

# 2. Calculer les matrices de covariance
cov_V = np.cov(V_centree, rowvar=False)
cov_W = np.cov(W_centree, rowvar=False)

# 3. Calcul des valeurs et vecteurs propres (étude spectrale)
valeurs_propres_V, vecteurs_propres_V = np.linalg.eig(cov_V)
valeurs_propres_W, vecteurs_propres_W = np.linalg.eig(cov_W)

# 4. Projection des individus sur la carte factorielle
V_proj = np.dot(V_centree, vecteurs_propres_V)
W_proj = np.dot(W_centree, vecteurs_propres_W)

# Affichage des résultats
print("Matrice V centrée :\n", V_centree)
print("\nMatrice W centrée :\n", W_centree)
print("\nMatrice de covariance de V :\n", cov_V)
print("\nMatrice de covariance de W :\n", cov_W)
print("\nValeurs propres de V :\n", valeurs_propres_V)
print("\nVecteurs propres de V :\n", vecteurs_propres_V)
print("\nProjection de V sur la carte factorielle :\n", V_proj)
print("\nProjection de W sur la carte factorielle :\n", W_proj)
