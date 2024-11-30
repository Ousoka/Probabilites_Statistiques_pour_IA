import numpy as np

#6 individus = p

# V = X - Xbar
V = np.array([
    [2, -3, -5],
    [-2, 2, 0],
    [0, 4, 2],
    [4, 0, 2],
    [2, -2, 0],
    [-6, -1, 1]
])

# Produit matriciel
tVV = np.dot(V.T, V) 

#Matrice variance covariance
M = 1/6*(tVV)

# print(tVV)

# print(M)

#Valeurs propores
# eigenvalues, _ = np.linalg.eig(M)

#Valeurs propores, vecteurs propres
eigenvalues, eigenvectors = np.linalg.eig(M)


# print(eigenvalues)

#axe1
#lambda1/sum(lambda)=(12/22)*100=54,54%

#axe1 et axe2
#(lambda1+lambda2)/sum(lambda)=(20/22)*100=90%

# Taches : trouver les vecteurs propres/coordonnees
#M*V=lambda*V
#(M-lambdaI)*V=0

#print(eigenvectors)


#matrice vecteurs propres

normalized_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

print(normalized_eigenvectors)