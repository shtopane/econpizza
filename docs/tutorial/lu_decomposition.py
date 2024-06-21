import scipy

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

def example1():
  A = np.array([[1, 2, 4], [3, 8, 14], [2, 6, 13]])
  sparse_A = scipy.sparse.csr_matrix(A)
  sparse_A_lu = scipy.sparse.linalg.splu(sparse_A)
  print(A)
  print(sparse_A)
  print(sparse_A_lu)
  # Partial pivoting, not the same as by hand
  print("L matrix:")
  print(sparse_A_lu.L.A)
  print("U matrix:")
  print(sparse_A_lu.U.A)

  print("Row permutation:")
  print(sparse_A_lu.perm_r)
  print("Column permutation:")
  print(sparse_A_lu.perm_c)

  lu_factor = sparse_A_lu.L.A - np.eye(len(sparse_A_lu.perm_r)) + sparse_A_lu.U.A

  print("Lu factor?: ", lu_factor)
  P, L, U = scipy.linalg.lu(A)
  print("L matrix:")
  print(L)
  print("U matrix:")
  print(U)

# example1()

def example2():
  from jax.experimental import sparse

  A = np.array([[1, 2, 4], [3, 8, 14], [2, 6, 13]])
  A_sp = sparse.BCSR(A, shape=A.shape)
  print(A_sp)

  A_lu = scipy.sparse.linalg.splu(A_sp)
  print(A_lu)

  pass
example2()