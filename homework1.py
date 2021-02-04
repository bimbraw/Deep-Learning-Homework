# %%
# Libraries
import numpy as np
# %%
# Q1-a
def problem_1a(A, B):
  if np.shape(A) != np.shape(B):
    print("Array shapes are not equal")
    return None
  else:
    A = np.array(A)
    B = np.array(B)
    return A + B

# %%
def problem_1b(A, B, C):
  # if np.shape(A)[1] != np.shape(B)[0]:
  #   print("Array dimensions don't match ")
  return np.dot(A, B) - C
# %%

def problem_1c(A, B, C):
  return np.array(A) * np.array(B) + np.transpose(C)

# %%
def problem_1d(x, y):
  return np.dot(np.transpose(x), y)

# %%
def problem_1e(A):
  return np.zeros(np.shape(A))

# %%
def problem_1f(A, x):
  return np.linalg.solve(A, x)

def problem_1g (A, x):
    A = np.array(A)
    x = np.array(x)
    return np.tranpose(np.linalg.solve(A.T, x.T))

def problem_1h (A, alpha):
    return np.array(A) + alpha*np.eye(np.shape(A))
# %%
def problem_1i (A, i):
  return np.sum(A[i, ::2])
    #return np.sum([arr[i][x] for x in range(len(arr[i])) if x%2 == 0])
# %%
def problem_1j (A, c, d):
    return np.mean(A[np.nonzero((A>=c) & (A<=d))])
# %%
def problem_1k (A, k):
    eig, vec = np.linalg.eig(A)
    k_eigvec = vec[eig.argsort()[::-1]][:k]
    return k_eigvec

def problem_1l (x, k, m, s):
    mean = x + m*np.ones(len(x))
    var = s*np.eye(len(x)) # Are we sure?
    return np.random.multivariate_normal(mean, var , size = (k,)).T
  # for n = 3 and k = 2, I got shape = (3, 2): Fixed.
# %%
def problem_1m (A):
    return np.random.permutation(A)

def problem_1n (x):
    y = (x - np.mean(x))/np.std(x)
    return y

def problem_1o (x, k):
    z = np.repeat(np.atleast_2d(x), k, axis = 0)
    return z
# %%
def problem_1p (X):
    x_3d = np.atleast_3d(X)
    x_1 = np.swapaxes(np.repeat(x_3d, 3, axis = 2), 1, 2)
    x_2 = np.swapaxes(np.swapaxes(np.repeat(x_3d, 3, axis = 2), axis1 = 0, axis2 = 2), axis1 = 1, axis2 = 2)
    return np.sqrt(np.sum((x_1 - x_2)**2, axis = 2))


# %%
def linear_regression (X_tr, y_tr):
  X_tr = X_tr.T
  xx = np.matmul(X_tr, X_tr.T)
  xy = np.matmul(X_tr, y_tr)
  return np.linalg.solve(xx, xy)

def fMSE(x, w, y):
  #print(w)
  x = x.T
  y_hat = np.matmul(x.T, w)
  cost = (np.sum(np.square(y_hat - y)))/(2*len(y))
  return cost

def train_age_regressor ():
    import numpy as np
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)
    # Report fMSE cost on the training and testing data (separately)
    print("Train FSME is: {:.4f}".format(fMSE(X_tr, w, ytr)))
    print("Test FSME is: {:.4f}".format(fMSE(X_te, w, yte)))
# %%
