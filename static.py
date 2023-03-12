import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from assembly import get_extended_system_beam, hermite_interpolator, L2_err

def exact_solution(q, L, support, Q_0 = 0, Q_L = 0, M_0 = 0, M_L = 0, a_0 = 0, a_L = 0, b_0 = 0):
    if isinstance(q, (int, float)):
        if support == "Cantilever":
            return lambda x: \
                (1/24)*x**2 * (6*(2*Q_L*L + 2*M_L + q*L**2) - 4*(Q_L + q*L)*x + q*x**2) + \
                a_0 + b_0*x
        elif support == "Simply supported":
            return lambda x: \
                (1/24)*(x/L) * (-4*(M_L + 2*M_0)*L**2 + q*L**4 + 12*M_0*L*x + 4*(M_L - M_0)*x**2 - 2*q*L**2*x**2 + q*L*x**3) + \
                a_0 + (x/L)*(a_L-a_0)
        else:
            raise Exception("Support not supported")
    else:
        raise Exception("Support not supported")                                                                

def test_cases(test_case):
    if  test_case == 1:
        q, L, a_0, b_0, Q_L, M_L = lambda _: 1, 1, 1, 1, 1, 1
        support = "Cantilever"

        fig, axs = plt.subplots(1, 2, figsize = (12.8, 4.8))

        n = 100
        x = np.linspace(0, L, n)
        S, _, fh = get_extended_system_beam(x, q, support, Q_L = Q_L, M_L = M_L, a_0 = a_0, b_0 = b_0)
        u_h = hermite_interpolator(x, spsolve(S.tocsc(), fh))
        
        mesh_refinements = np.linspace(2, 1000, 20, dtype = np.int16)
        error_refinements = np.zeros(len(mesh_refinements))
        u = exact_solution(q(0), L, support, Q_L = Q_L, M_L = M_L, a_0 = a_0, b_0 = b_0)

        for i in range(len(mesh_refinements)):
            mesh = np.linspace(0, L, mesh_refinements[i])
            S_i, _, fh_i = get_extended_system_beam(mesh, q, support, Q_L = Q_L, M_L = M_L, a_0 = a_0, b_0 = b_0)
            u_h_i = hermite_interpolator(mesh, spsolve(S_i.tocsc(), fh_i))
            error_refinements[i] = L2_err(u, u_h_i)
        
        axs[0].plot(x, u_h(x), label = 'numerical solution')
        axs[0].plot(x, u(x), '--', label = 'exact solution')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('w')
        
        h = L/np.linspace(2, 1000, 20)
        axs[1].loglog(h, error_refinements, label = r'$L^2$ error')
        axs[1].loglog(h, error_refinements[0]*(h/h[0])**4, 'k--', label = r'$h^{-4}$')
        axs[1].set_xlabel('h')

        fig.suptitle('Cantilever beam')
        return fig, axs
    elif test_case == 2:
        q, L, a_0, a_L, M_0, M_L = lambda _: 1, 1, 1, 1, 1, 1
        support = "Simply supported"

        fig, axs = plt.subplots(1, 2, figsize = (12.8, 4.8))

        n = 100
        x = np.linspace(0, L, n)
        S, _, fh = get_extended_system_beam(x, q, support, M_0 = M_0, M_L = M_L, a_0 = a_0, a_L = a_L)
        u_h = hermite_interpolator(x, spsolve(S.tocsc(), fh))
        
        mesh_refinements = np.linspace(2, 1000, 20, dtype = np.int16)
        error_refinements = np.zeros(len(mesh_refinements))
        u = exact_solution(q(0), L, support, M_0 = M_0, M_L = M_L, a_0 = a_0, a_L = a_L)

        for i in range(len(mesh_refinements)):
            mesh = np.linspace(0, L, mesh_refinements[i])
            S_i, _, fh_i = get_extended_system_beam(mesh, q, support, M_0 = M_0, M_L = M_L, a_0 = a_0, a_L = a_L)
            u_h_i = hermite_interpolator(mesh, spsolve(S_i.tocsc(), fh_i))
            error_refinements[i] = L2_err(u, u_h_i)
        
        axs[0].plot(x, u_h(x), label = 'numerical solution')
        axs[0].plot(x, u(x), '--', label = 'exact solution')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('w')
        
        h = L/np.linspace(2, 1000, 20)
        axs[1].loglog(h, error_refinements, label = r'$L^2$ error')
        axs[1].loglog(h, error_refinements[0]*(h/h[0])**4, 'k--', label = r'$h^{-4}$')
        axs[1].set_xlabel('h')

        fig.suptitle('Simply supported beam')
        return fig, axs
    else:
        raise Exception("Test case not implemented!")

fig, axs = test_cases(2)
axs[0].legend()
axs[1].legend()
plt.show()


