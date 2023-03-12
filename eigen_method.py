import numpy as np
from scipy.integrate import quad
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
import matplotlib.pyplot as plt
from bisect import bisect
from scipy.optimize import fsolve
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
from assembly import get_extended_system_beam, hermite_interpolator, L2_err, get_global_matrices

u = lambda x: 1/24 * x**2 *(6 - 4*x + x**2)
         
plt.figure()
n = 20
mesh = np.linspace(0, 1, n)
x = np.linspace(0, 1, 10000)
S_i, _, fh_i = get_extended_system_beam(mesh, lambda _: 1, "Cantilever", M_0 = 0, M_L = 0, a_0 = 0, a_L = 0)
u_h = hermite_interpolator(mesh, spsolve(S_i.tocsc(), fh_i))
   
print(L2_err(u, u_h))
plt.plot(x, [u_h(xi) for xi in x], label = 'numerical solution')
plt.plot(x, u(x), '--', label = 'exact solution')
plt.title('n = {n}'.format(n = 1000))
plt.legend()
plt.show()

'''
Compute A and its eigenvalues, eigenvectors
'''
Se, Me, f = get_extended_system_beam(mesh, lambda _: 0, "Cantilever", M_0 = 0, M_L = 0, a_0 = 0, a_L = 0)
A = inv(Se.tocsc()) @ Me.tocsc()
eigval, eigvec = np.linalg.eig(A.toarray())

# Sort eigenvalues and corresponding eigenvectors
omega_k = 1/np.sqrt(eigval[2:19])  # eigen frequencies omega_k
idx = omega_k.argsort()[::1]
omega_k = omega_k[idx]
eigvec = eigvec[idx]

'''
Function to get analytic eigen frequencies
'''
def getroots():
    f = lambda x: 1 + np.cosh(x)*np.cos(x)
    '''
    The reason why we separate the initial guess array 
    is that when it comes to the interval [20,50], we 
    have to change the step size of initial guessing,
    or the result does not converge.
    '''
    inv1 = list(np.arange(2,30,3))
    inv2 = list(np.arange(30,50,3.5))
    inv3 = list(np.arange(50,60,3))
    x1 = fsolve(f,inv1)
    x2 = fsolve(f,inv2)[1:]
    x3 = fsolve(f,inv3)
    
    return np.concatenate((x1,x2,x3),axis=0)

'''Analytic eigenfrequencies'''
omega_j = getroots()
omega_j = omega_j**2  # analytic eigenfrequencies omega_j

plt.figure()
plt.plot(np.linspace(1,len(omega_k),len(omega_k)),omega_k,c='black',label='numerical eigenfrequencies')
plt.plot(np.linspace(1,len(omega_j),len(omega_j)),omega_j,c='black',label='analytic eigenfrequencies',linestyle='--')
plt.legend()
plt.ylabel('eigenfrequencies')
plt.grid(linestyle=':')
plt.xlabel('n=10')
plt.show()

'''Analytic eigenfunctions'''
k_list = [1.8751,4.6941,7.8548,10.996,14.137]
x = np.linspace(0,1,100)
fig,ax = plt.subplots(2,2,figsize=[20,15])
plt.suptitle('First Four Eigenfunctions')
index = 0
for i in range(2):
    for j in range(2):
        k = k_list[index]
        y = np.cosh(k*x)-np.cos(k*x)-(np.cosh(k)+np.cos(k))*(np.sinh(k*x)-np.sin(k*x))/(np.sinh(k)+np.sin(k))
        ax[i][j].plot(x,y)
        ax[i][j].set_title('w{}'.format(index+1))
        index += 1
plt.show()

'''
function to derive w(t) by eigen stuff
'''
def eigenmodes( w0,wd0,Me,Se,mesh,x ):
    A = inv(Se.tocsc()) @ Me.tocsc()
    eigval, eigvec = np.linalg.eig(A.toarray())
    idx0 = np.argwhere( eigval==0 ) # to get rid of 0 eigenvalues
    eigval = np.delete(eigval,idx0)
    eigvec = eigvec.T
    eigvec = np.delete(eigvec,idx0,axis=0)
    omega_k = 1/np.sqrt(eigval)
    
    def u_t(t):
        superposition = np.zeros(Me.shape[0])
        for i in range( len(omega_k) ):
            alpha_k = (eigvec[i][:-2] @ Me[:-2,:-2] @ w0) / (eigvec[i][:-2] @ Me[:-2,:-2] @ eigvec[i][:-2])
            beta_k = (eigvec[i][:-2] @ Me[:-2,:-2] @ wd0) / (eigvec[i][:-2] @ Me[:-2,:-2] @ eigvec[i][:-2])
            coeff = alpha_k*np.cos(omega_k[i]*t) + beta_k*np.sin(omega_k[i]*t)/omega_k[i]
            superposition += (coeff * eigvec[i])
        u_h = hermite_interpolator(mesh, superposition[:-2])
        u_h_array = [u_h(xi) for xi in x]
        return u_h_array
    
    return u_t

w0 = np.array([u_h(xi) for xi in np.linspace(0,1,2*n)])
wd0 = np.zeros(2*n)
mesh = np.linspace(0, 1, n)
x = np.linspace(0, 1, 100)
u_t = eigenmodes( w0,wd0,Me,Se,mesh,x )
# plt.plot(x, u_t(1), label = 'numerical solution')

'''
Generate a movie
'''
duration = 2
fig_mpl, ax = plt.subplots(1,figsize=(5,3), facecolor='white')
ax.set_ylim(-0.2,0.2)
line, = ax.plot(x, u_t(0), lw=2,c='black')
ax.set_title('beam')
ax.set_xlabel('L')
ax.grid(linestyle=':')
def make_frame_mpl(t):
    line.set_ydata( u_t(t))
    return mplfig_to_npimage(fig_mpl)

animation =mpy.VideoClip(make_frame_mpl, duration=duration)
animation.write_gif("sinc_mpl.gif", fps=30)
