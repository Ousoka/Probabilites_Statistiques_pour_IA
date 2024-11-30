import numpy as np

V = np.array([
    [8, 1, 0],
    [4, 6, 5],
    [6, 8, 7],
    [10, 4, 7],
    [8, 2, 5],
    [0, 3, 6]
])

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
C1 = np.dot(V, U1)
C2 = np.dot(V, U2)

# Printing results as pairs of coordinates
print("Les nouvelles coordonnées:")
for i in range(len(C1)):
    print(f"I{i+1} = ({C1[i, 0]:.2f}, {C2[i, 0]:.2f})")
