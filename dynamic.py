import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import splu, spsolve
from matplotlib.animation import FuncAnimation
from assembly import get_extended_system_beam, hermite_interpolator

def Newmark(M, S, fh, K, T):
    dt = T/K
    u = np.zeros((K+1, len(fh)))
    fh_no_load = np.zeros(len(fh))
    fh_no_load[-2:] = fh[-2:]
    u[0, :] = spsolve(S.tocsc(), fh)
    u_star = u
    v, v_star, a = (0, 0, 0)
    lu = splu(M + 0.25*dt**2*S)
    for i in range(K):
        u_star = u[i, :] + dt*v + 0.25*dt**2*a
        v_star = v + 0.5*dt*a
        a = lu.solve(fh_no_load - S*u_star)
        v = v_star + 0.5*dt*a
        u[i+1, :] = u_star + 0.25*dt**2*a
    return u

def create_video(T, m, n, q, support, Q_0 = 0, Q_L = 0, M_0 = 0, M_L = 0, a_0 = 0, a_L = 0, b_0 = 0, b_L = 0, ylim = (-0.2, 0.2)):
    fig = plt.figure()
    ax = plt.axes(xlim = (0, 1.2), ylim = ylim)
    line, = ax.plot([], [], lw = 3) 

    mesh = np.linspace(0, 1, n)
    S, M, fh = get_extended_system_beam(mesh, q, support, Q_0 = Q_0, Q_L = Q_L, M_0 = M_0, M_L = M_L, a_0 = a_0, a_L = a_L, b_0 = b_0, b_L = b_L)
    u_h = Newmark(M.tocsc(), S.tocsc(), fh, m, T)

    def animate(i):
        x = np.linspace(0, 1, 100)
        solution = hermite_interpolator(mesh, u_h[i, :])
        y = solution(x)
        line.set_data(x, y)
        return line,
    
    anim = FuncAnimation(fig, animate, frames = m, interval = round(T/m*1000), repeat_delay = 1000)
    plt.show()
    return anim

T = 100
m = 2000
n = 100
q = lambda _: 1
support = "Cantilever"
anim = create_video(T, m, n, q, support, Q_L = 0.01, M_L = 0, a_0 = 0, b_0 = 0, ylim = (-0.2, 0.2))

