import numpy as np
from scipy.integrate import quad, fixed_quad
from scipy.sparse import dok_matrix
from bisect import bisect

def get_local_matrices(h, q):
    phi_scaled = [lambda x: 1 - 3*x**2 + 2*x**3, lambda x: h*x*(x-1)**2, lambda x: 3*x**2 - 2*x**3, lambda x: h*x**2*(x-1)]
    d2phi_scaled = [lambda x: -6 + 12*x, lambda x: h*(4*(x - 1) + 2*x), lambda x: 6 - 12*x, lambda x: h*(2*(x - 1) + 4*x)]
    S = np.zeros((4, 4))
    M = np.zeros((4, 4))
    fh = np.zeros(4)
    for i in range(4):
        fh[i] = h * quad(lambda x: phi_scaled[i](x) * q(x), 0, 1)[0]
        for j in range(4):
            S[i, j] = 1/h**3 * fixed_quad(lambda x: d2phi_scaled[i](x) * d2phi_scaled[j](x), 0, 1, n = 2)[0]
            M[i, j] = h * fixed_quad(lambda x: phi_scaled[i](x) * phi_scaled[j](x), 0, 1, n = 4)[0]
    return S, M, fh

def get_global_matrices(mesh, q):
    n = len(mesh) - 1
    S = dok_matrix((2*(n+1), 2*(n+1)))
    M = dok_matrix((2*(n+1), 2*(n+1)))
    fh = np.zeros(2*(n+1))
    for el in range(n):
        h = mesh[el+1] - mesh[el]
        S_local, M_local, fh_local = get_local_matrices(h, lambda x: q((x - mesh[el])/h))
        for i in range(4):
            fh[2*el + i] += fh_local[i]
            for j in range(4):
                S[2*el + i, 2*el + j] += S_local[i, j]
                M[2*el + i, 2*el + j] += M_local[i, j]
    return S, M, fh

def get_extended_system_beam(mesh, q, support, Q_0 = 0, Q_L = 0, M_0 = 0, M_L = 0, a_0 = 0, a_L = 0, b_0 = 0, b_L = 0):
    n = len(mesh) - 1
    fh = np.zeros(2*(n+1)+2)
    S, M, fh[:-2] = get_global_matrices(mesh, q)
    S.resize((2*(n+1)+2, 2*(n+1)+2))
    M.resize((2*(n+1)+2, 2*(n+1)+2))
    if support == "Cantilever":
        # Extended stiffness matrix
        S[0, 2*(n+1)] = 1
        S[1, 2*(n+1) + 1] = 1
        S[2*(n+1), 0] = 1
        S[2*(n+1) + 1, 1] = 1
        # Right-hand side
        fh[2*(n+1)] = a_0
        fh[2*(n+1) + 1] = b_0
        fh[2*n] += Q_L
        fh[2*n + 1] += M_L
    elif support == "Simply supported":
        # Extended stiffness_matrix
        S[0, 2*(n+1)] = 1
        S[2*n, 2*(n+1) + 1] = -1
        S[2*(n+1), 0] = 1
        S[2*(n+1) + 1, 2*n] = -1
        # Right-hand side
        fh[1] -= M_0
        fh[2*n + 1] += M_L
        fh[2*(n+1)] = a_0
        fh[2*(n+1) + 1] = -a_L
    else:
        raise Exception("Support not supported")
        
    return S, M, fh

def langrangian_interpolator(mesh, coeff):
    def I(x):
        n = len(mesh) - 1
        el = min(bisect(mesh, x) - 1, n-1)  # find corresponding element
        h = mesh[el+1] - mesh[el] # calculate element size
        t = (x - mesh[el])/h # transform to coordinates on reference element [0, 1]

        loc_dof = coeff[el:el+2]
        loc_shape = [1 - t, t]
        return np.dot(loc_dof, loc_shape)
    return np.vectorize(I)

def hermite_interpolator(mesh, coeff):
    def I(x):
        n = len(mesh) - 1
        el = min(bisect(mesh, x) - 1, n-1)  # find corresponding element
        h = mesh[el+1] - mesh[el] # calculate element size
        t = (x - mesh[el])/h # transform to coordinates on reference element [0, 1]

        loc_dof = coeff[2*el:2*(el+2)]
        loc_shape = [1 - 3*t**2 + 2*t**3, h*t*(t-1)**2, 3*t**2 - 2*t**3, h*t**2*(t-1)]
        return np.dot(loc_dof, loc_shape)
    return np.vectorize(I)

def L2_err(u, u_h):
    return np.sqrt(quad(lambda x: (u(x) - u_h(x))**2, 0, 1)[0])
