import numpy as np
import matplotlib.pyplot as plt

def center_data(X):
    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    centered_X = X - mean
    return centered_X

def compute_covariance_matrix(X):
    # Step 2: Calculate the covariance matrix
    N = X.shape[0]
    covariance_matrix = np.dot(X.T, X) / N
    return covariance_matrix

def compute_metric_matrix(X, homogeneous=True):
    # Step 3: Define the metric matrix M
    if homogeneous:
        # For homogeneous data, M is the identity matrix
        M = np.identity(X.shape[1])
    else:
        # For heterogeneous data, M is a diagonal matrix with 1/variance for each feature
        variances = np.var(X, axis=0)
        M = np.diag(1 / variances)
    return M

def normalize_matrix(M):
    # Calculate the Frobenius norm of M
    norm_M = np.linalg.norm(M, 'fro')
    
    # Normalize M by dividing each element by the Frobenius norm
    M_normalized = M / norm_M if norm_M != 0 else M  # Avoid division by zero
    return M_normalized

def normalize_eigenvectors(eigenvectors, M):
    # Normalize eigenvectors with respect to M
    normalized_eigenvectors = []
    for u in eigenvectors.T:  # each column of eigenvectors
        norm_factor = np.dot(u.T, np.dot(M, u)) ** 0.5
        normalized_eigenvectors.append(u / norm_factor)
    return np.array(normalized_eigenvectors).T

def pca_with_metric(X, homogeneous=True, threshold=0.80):
    # Step 1: Center the data
    centered_X = X - np.mean(X, axis=0)
    
    # Step 2: Compute covariance matrix
    N = centered_X.shape[0]
    covariance_matrix = np.dot(centered_X.T, centered_X) / N
    
    # Step 3: Compute the metric matrix M
    M = compute_metric_matrix(centered_X, homogeneous=homogeneous)
    
    # Step 4: Eigen decomposition of VM
    VM = np.dot(covariance_matrix, M)
    eigenvalues, eigenvectors = np.linalg.eig(VM)
    
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Normalize eigenvectors with respect to M
    eigenvectors = normalize_eigenvectors(eigenvectors, M)
    
    # Step 6: Calculate cumulative variance
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_components = np.where(cumulative_variance >= threshold)[0][0] + 1
    
    # Step 7: Calculate principal components
    principal_components = np.dot(centered_X, eigenvectors)
    
    return eigenvalues, eigenvectors, cumulative_variance, num_components, principal_components


V = np.array([
    [8, 1, 0],
    [4, 6, 5],
    [6, 8, 7],
    [10, 4, 7],
    [8, 2, 5],
    [0, 3, 6]
])

#matrice centrée
C=center_data(V)
# print(C)

#matrice variance covariance
VCov = compute_covariance_matrix(C)
# print(VCov)

#Valeurs propres

#Vecteurs propres

#Normes Vecteurs propres

#Coordonnees des individus dans le nouveau repère

U1 = np.array([
    [0.816],
    [-0.416],
    [-0.416]
])

U2 = np.array([
    [-0.577],
    [-0.577],
    [-0.577]
])

# Calculating projections onto U1 and U2
C1 = np.dot(C, U1)
C2 = np.dot(C, U2)

# Printing results as pairs of coordinates
print("Les nouvelles coordonnées:")
for i in range(len(C1)):
    print(f"I{i+1} = ({C1[i, 0]:.2f}, {C2[i, 0]:.2f})")

#Contribution

