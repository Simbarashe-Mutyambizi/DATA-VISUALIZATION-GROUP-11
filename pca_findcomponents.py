from numpy import array, mean, cov
from numpy.linalg import eig

# Step 1
M = array([
    [1, 3],
    [2, 2],
    [3, 1]
    ])

# Step 2
# compute a vector of means of each column
means_vec = mean(M, axis=0)

# canculated the new centred matrix
# Numpy applies broadcasting to reshape means_vec to the size of matrix M
# Ref: https://numpy.org/devdocs/user/basics.broadcasting.html 
C = M - means_vec


# Step 3
# calculate covariance matrix of the centred matrix
K = cov(C.T)


# Step 4
# find the eigenvectors and eigenvalues from the covariance matrix
eigvals, eigvecs = eig(K)

# show all eigenvectors and eigenvalues
print("Eigenvectors (Q):\n", eigvecs)
print("Eigenvalues (Lambdas):\n", eigvals)

# print individual principal components
print("Principal Component 1:")
print("\tEigenvector:", eigvecs[:,0])
print("\tEigenvalue:", eigvals[0])

print("Principal Component 2:")
print("\tEigenvector:", eigvecs[:,1])
print("\tEigenvalue:", eigvals[1])