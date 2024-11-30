import numpy as np

# 6 individus = p
# Matrice centrée (X - Xbar)
V = np.array([
    [2, -3, -5],
    [-2, 2, 0],
    [0, 4, 2],
    [4, 0, 2],
    [2, -2, 0],
    [-6, -1, 1]
])

# Matrix of poids and taille for six individuals
W = np.array([
    [20, 15],  # I1
    [5, 2],    # I2
    [12, 21],  # I3
    [21, 13],  # I4
    [2, 7],    # I5
    [12, 20]   # I6
])

# Produit matriciel
tVV = np.dot(V.T, V) 

tWW = np.dot(W.T, W) 

# Matrice variance covariance
M = 1/5 * tVV  # Change 1/6 to 1/5 for unbiased covariance
N = 1/5 * tWW

# Valeurs propres et vecteurs propres
eigenvaluesM, eigenvectorsM = np.linalg.eig(M)
eigenvaluesN, eigenvectorsN = np.linalg.eig(N)

print("Eigenvalues of M:", eigenvaluesM)
print("Eigenvectors of M:", eigenvectorsM)

print("Eigenvalues of N:", eigenvaluesN)
print("Eigenvectors of N:", eigenvectorsN)


# Affichage des valeurs et vecteurs propres
print("Valeurs propres M:")
print(eigenvaluesM)

print("\nVecteurs propres M:")
print(eigenvectorsM)

# print("Valeurs propres N:")
print(eigenvaluesN)

print("\nVecteurs propres N:")
print(eigenvectorsN)

# Proportion de variance expliquée
variance_explainedM = eigenvaluesM / np.sum(eigenvaluesM) * 100
print("\nProportion de variance expliquée par chaque axe :")
print(variance_explainedM)

# Projection des données sur les composantes principales
# Normaliser les vecteurs propres
eigenvectors_normalizedM = eigenvectorsM / np.linalg.norm(eigenvectorsM, axis=0)

# Projeter les données sur les vecteurs propres
projected_dataM = np.dot(V, eigenvectors_normalizedM)

print("\nDonnées projetées sur les axes principaux :")
print(projected_dataM)

# Visualisation des projections (facultatif)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.scatter(projected_dataM[:, 0], projected_dataM[:, 1], marker='o')
plt.title('Projection des données sur les axes principaux')
plt.xlabel('Axe Principal 1')
plt.ylabel('Axe Principal 2')
plt.grid()
plt.show()
