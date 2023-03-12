import numpy as np
from scipy.integrate import fixed_quad, quad
from bisect import bisect
from scipy.sparse import block_diag, dok_matrix, bmat
from scipy.sparse.linalg import splu, spsolve, eigs
from numpy.linalg import norm

class Beam:
    def __init__(self, position, angle, length, mesh_size, physical_parameters, \
         initial_load_longitudinal = lambda _: 0, initial_load_transversal = lambda _: 0):

        self.position = position
        self.angle = angle
        self.length = length
        self.modulus = physical_parameters['modulus']
        self.mass = physical_parameters['density'] * physical_parameters['area']
        self.inertia = (1/12) * physical_parameters['area']**2
        self.area = physical_parameters['area']
        self.mesh_size = mesh_size
        if isinstance(initial_load_longitudinal, (int, float)):
            self.initial_load_longitudinal = lambda _: initial_load_longitudinal
        else:
            self.initial_load_longitudinal = initial_load_longitudinal
        if isinstance(initial_load_transversal, (int, float)):
            self.initial_load_transversal = lambda _: initial_load_transversal
        else:
            self.initial_load_transversal = initial_load_transversal
        self.mesh = np.linspace(0, length, self.mesh_size + 1)

    def get_local_stiffness_matrix(self, h):
        # E and I are assumed to be constant
        dphi = [-1, 1]
        d2phi_scaled = [lambda x: -6 + 12*x, lambda x: h*(4*(x - 1) + 2*x), lambda x: 6 - 12*x, lambda x: h*(2*(x - 1) + 4*x)]
        S_L = np.zeros((2, 2))
        S_T = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                S_L[i, j] = 1/h * dphi[i]*dphi[j]
        for i in range(4):
            for j in range(4):
                S_T[i, j] = 1/h**3 * \
                    fixed_quad(lambda x: d2phi_scaled[i](x) * d2phi_scaled[j](x), 0, 1, n = 2)[0]
        return S_L, S_T

    def get_local_mass_matrix(self, h):
        # mu is assumed to be constant
        phi = [lambda x: 1-x, lambda x: x]
        phi_scaled = [lambda x: 1 - 3*x**2 + 2*x**3, lambda x: h*x*(x-1)**2, lambda x: 3*x**2 - 2*x**3, lambda x: h*x**2*(x-1)]
        M_L = np.zeros((2, 2))
        M_T = np.zeros((4, 4))
        for i in range(2):
            for j in range(2):
                M_L[i, j] = self.mass/(self.modulus * self.area) * h * fixed_quad(lambda x: phi[i](x) * phi[j](x), 0, 1, n = 2)[0]
        for i in range(4):
            for j in range(4):
                M_T[i, j] = self.mass/(self.modulus * self.inertia) * h * fixed_quad(lambda x: phi_scaled[i](x) * phi_scaled[j](x), 0, 1, n = 4)[0]
        return M_L, M_T

    def get_local_rhs(self, h):
        # f is assumed to be independent of time
        phi = [lambda x: 1-x, lambda x: x]
        phi_scaled = [lambda x: 1 - 3*x**2 + 2*x**3, lambda x: h*x*(x-1)**2, lambda x: 3*x**2 - 2*x**3, lambda x: h*x**2*(x-1)]
        fh_L = np.zeros(2)
        fh_T = np.zeros(4)
        for i in range(2):
            fh_L[i] = 1/(self.modulus * self.area) * h * quad(lambda x: phi[i](x) * self.initial_load_longitudinal(x), 0, 1)[0]
        for i in range(4):
            fh_T[i] = 1/ (self.modulus * self.inertia) * h * quad(lambda x: phi_scaled[i](x) * self.initial_load_transversal(x), 0, 1)[0]
        return fh_L, fh_T

    def get_global_stiffness_matrix(self):
        n = self.mesh_size
        S_L = dok_matrix((n+1, n+1))
        S_T = dok_matrix((2*(n+1), 2*(n+1)))
        for el in range(n):
            h = self.mesh[el+1] - self.mesh[el]
            S_L_local, S_T_local = self.get_local_stiffness_matrix(h)
            for i in range(2):
                for j in range(2):
                    S_L[el + i, el + j] += S_L_local[i, j]
            for i in range(4):
                for j in range(4):
                    S_T[2*el + i, 2*el + j] += S_T_local[i, j]
        return S_L, S_T

    def get_global_mass_matrix(self):
        n = self.mesh_size
        M_L = dok_matrix((n+1, n+1))
        M_T = dok_matrix((2*(n+1), 2*(n+1)))
        for el in range(n):
            h = self.mesh[el+1] - self.mesh[el]
            M_L_local, M_T_local = self.get_local_mass_matrix(h)
            for i in range(2):
                for j in range(2):
                    M_L[el + i, el + j] += M_L_local[i, j]
            for i in range(4):
                for j in range(4):
                    M_T[2*el + i, 2*el + j] += M_T_local[i, j]
        return M_L, M_T

    def get_global_rhs(self):
        n = self.mesh_size
        fh_L = np.zeros(n+1)
        fh_T = np.zeros(2*(n+1))
        for el in range(n):
            h = self.mesh[el+1] - self.mesh[el]
            fh_L_local, fh_T_local = self.get_local_rhs(h)
            for i in range(2):
                fh_L[el + i] += fh_L_local[i]
            for i in range(4):
                fh_T[2*el + i] += fh_T_local[i]
        return fh_L, fh_T

    def langrangian_interpolator(self, coeff):
        def I(x):
            n = self.mesh_size
            el = min(bisect(self.mesh, x) - 1, n-1)  # find corresponding element
            h = self.mesh[el+1] - self.mesh[el] # calculate element size
            t = (x - self.mesh[el])/h # transform to coordinates on reference element [0, 1]
            loc_dof = coeff[el:el+2]
            loc_shape = [1 - t, t]
            return np.dot(loc_dof, loc_shape)
        return np.vectorize(I)

    def hermitian_interpolator(self, coeff):
        def I(x):
            n = self.mesh_size
            el = min(bisect(self.mesh, x) - 1, n-1)  # find corresponding element
            h = self.mesh[el+1] - self.mesh[el] # calculate element size
            t = (x - self.mesh[el])/h # transform to coordinates on reference element [0, 1]
            loc_dof = coeff[2*el:2*(el+2)]
            loc_shape = [1 - 3*t**2 + 2*t**3, h*t*(t-1)**2, 3*t**2 - 2*t**3, h*t**2*(t-1)]
            return np.dot(loc_dof, loc_shape)
        return np.vectorize(I)

    def get_results(self, coeff, grid_size):
        v = self.langrangian_interpolator(coeff[:self.mesh_size+1])
        w = self.hermitian_interpolator(coeff[self.mesh_size+1:])
        grid = np.linspace(0, self.length, grid_size)
        x = self.position[0] + np.cos(self.angle)*(grid + v(grid)) - np.sin(self.angle)*w(grid)
        y = self.position[1] + np.sin(self.angle)*(grid + v(grid)) + np.cos(self.angle)*w(grid)
        return x, y

class Framework:
    def __init__(self):
        self.beam_list = []
        self.total_time_steps = 0
        self.constraint_matrix = dok_matrix((0, 0))
    
    def add_beam(self, beam):
        self.beam_list.append(beam)

    def add_constraint(self, constraint_type, beams, constraint_side):
        index = [sum([3*(beam.mesh_size + 1) for beam in self.beam_list[:beams[i]-1]]) for i in range(len(beams))]
        C = self.constraint_matrix
        if constraint_type == "linking":
            C.resize((sum([3*(beam.mesh_size + 1) for beam in self.beam_list]), C.shape[1]+2))
            n = [(self.beam_list[beams[0]-1]).mesh_size, (self.beam_list[beams[1]-1]).mesh_size]
            phi = [(self.beam_list[beams[0]-1]).angle, (self.beam_list[beams[1]-1]).angle]
            for i in range(2):
                if constraint_side[i] == "left":
                    C[index[i],                         -2] = (-1)**i * np.cos(phi[i])
                    C[index[i] + (n[i]+1),              -2] = (-1)**i * -np.sin(phi[i])
                    C[index[i],                         -1] = (-1)**i * np.sin(phi[i])
                    C[index[i] + (n[i]+1),              -1] = (-1)**i * np.cos(phi[i])
                else:
                    C[index[i] + n[i],                  -2] = (-1)**i * np.cos(phi[i])
                    C[index[i] + (n[i]+1) + 2*n[i],     -2] = (-1)**i * -np.sin(phi[i])
                    C[index[i] + n[i],                  -1] = (-1)**i * np.sin(phi[i]) 
                    C[index[i] + (n[i]+1) + 2*n[i],     -1] = (-1)**i * np.cos(phi[i])
        elif constraint_type == "fixed_bearing":
            C.resize((sum([3*(beam.mesh_size + 1) for beam in self.beam_list]), C.shape[1]+3))
            n = (self.beam_list[beams[0]-1]).mesh_size
            if constraint_side[0] == "left":
                C[index[0],                             -3] = 1
                C[index[0] + (n+1),                     -2] = 1
                C[index[0] + (n+1) + 1,                 -1] = 1
            else:
                C[index[0] + n,                         -3] = 1
                C[index[0] + (n+1) + 2*n,               -2] = 1
                C[index[0] + (n+1) + (2*n + 1),         -1] = 1
        elif constraint_type == "stiffness_of_angles":
            C.resize((sum([3*(beam.mesh_size + 1) for beam in self.beam_list]), C.shape[1]+1))
            n = [(self.beam_list[beams[0]-1]).mesh_size, (self.beam_list[beams[1]-1]).mesh_size]
            for i in range(2):
                if constraint_side[i] == "left":
                    C[index[i] + (n[i]+1) + 1,          -1] = (-1)**i
                else:
                    C[index[i] + (n[i]+1) + (2*n[i]+1), -1] = (-1)**i
        elif constraint_type == "movable_bearing_y":
            C.resize((sum([3*(beam.mesh_size + 1) for beam in self.beam_list]), C.shape[1]+1))
            n = (self.beam_list[beams[0]-1]).mesh_size
            phi = (self.beam_list[beams[0]-1]).angle
            if constraint_side[0] == "left":
                C[index[0],                             -1] = np.cos(phi)
                C[index[0] + 1,                         -1] = -np.sin(phi)
            else:
                C[index[0] + n,                         -1] = np.cos(phi)
                C[index[0] + (n+1) + 2*n,               -1] = -np.sin(phi)
        elif constraint_type == "movable_bearing_x":
            C.resize((sum([3*(beam.mesh_size + 1) for beam in self.beam_list]), C.shape[1]+1))
            n = (self.beam_list[beams[0]-1]).mesh_size
            phi = (self.beam_list[beams[0]-1]).angle
            if constraint_side[0] == "left":
                C[index[0],                             -1] = np.sin(phi)
                C[index[0] + 1,                         -1] = np.cos(phi)
            else:
                C[index[0] + n,                         -1] = np.sin(phi)
                C[index[0] + (n+1) + 2*n,               -1] = np.cos(phi)
        else:
            raise Exception("Constraint not impelemented yet!")
        self.constraint_matrix = C
        return None

    def get_system_stiffness_matrix(self):
        stiffness_matrices = []
        for beam in self.beam_list:
            S_L, S_T = beam.get_global_stiffness_matrix()
            stiffness_matrices.append(S_L)
            stiffness_matrices.append(S_T)
        return block_diag(stiffness_matrices)

    def get_system_mass_matrix(self):
        mass_matrices = []
        for beam in self.beam_list:
            M_L, M_T = beam.get_global_mass_matrix()
            mass_matrices.append(M_L)
            mass_matrices.append(M_T)
        return block_diag(mass_matrices)

    def get_system_rhs(self):
        rhs_vectors = []
        for beam in self.beam_list:
            fh_L, fh_T = beam.get_global_rhs()
            rhs_vectors.append(fh_L)
            rhs_vectors.append(fh_T)
        return np.concatenate(rhs_vectors)
        
    def get_extended_stiffness_matrix(self):
        S = self.get_system_stiffness_matrix()
        C = self.constraint_matrix
        return bmat([[S, C], [C.transpose(), None]])

    def get_extended_mass_matrix(self):
        M = self.get_system_mass_matrix()
        M.resize((M.shape[0] + (self.constraint_matrix).shape[1], M.shape[0] + (self.constraint_matrix).shape[1]))
        return M

    def get_extended_rhs(self):
        # Currently f is assumed to be independent of time
        f = self.get_system_rhs()
        ftilde = np.zeros(len(f) + (self.constraint_matrix).shape[1])
        ftilde[:len(f)] = f
        # The a(t) 'vector of values of constraints' should still be added here
        return ftilde

    def static_solve(self):
        S = self.get_extended_stiffness_matrix()
        ftilde = self.get_extended_rhs()
        self.total_time_steps = 1
        solution = spsolve(S.tocsc(), ftilde)
        solution.resize(1, len(ftilde))
        return solution

    def dynamic_solve(self, initial_condition, dt, terminal_time, fps):
        # Left hand side is assumed to be independent of time
        self.total_time_steps = round(terminal_time/dt) + 1
        S = (self.get_extended_stiffness_matrix()).tocsc()
        M = (self.get_extended_mass_matrix()).tocsc()
        ftilde = self.get_extended_rhs()
        ftilde_no_load = np.zeros(len(ftilde))
        u = np.zeros((self.total_time_steps, len(ftilde)))
        u[0, :] = initial_condition
        u_star = u
        v, v_star, a = (0, 0, 0)
        lu = splu(M + 0.25*dt**2*S)
        frame_count = 0
        temp = [initial_condition]
        time = [0]
        for i in range(self.total_time_steps - 1):
            u_star = u[i, :] + dt*v + 0.25*dt**2*a
            v_star = v + 0.5*dt*a
            a = lu.solve(ftilde_no_load - S*u_star)
            v = v_star + 0.5*dt*a
            u[i+1, :] = u_star + 0.25*dt**2*a
            if (i+1)*dt >= frame_count/fps:
                temp.append(u_star + 0.25*dt**2*a)
                time.append((i+1)*dt)  
                frame_count += 1
        results = np.zeros((len(temp), M.shape[0]))
        for i in range(len(temp)):
            results[i, :] = temp[i]
        return time, results
    
    def eigen_solve(self, initial_condition, number_of_eig, dt, terminal_time, fps):
        self.total_time_steps = round(terminal_time/dt) + 1
        Se = self.get_extended_stiffness_matrix()
        Me = self.get_extended_mass_matrix()
        M = self.get_system_mass_matrix()
        number_of_constraints = Me.shape[0] - M.shape[0]
        eigval, eigvec = eigs(Me, M = Se, k = number_of_eig, which = 'LM')
        idx0 = np.where(eigval != 0)
        eigval = np.reshape(np.array(np.real(eigval)[idx0[0]]), [1, len(eigval)])
        eigvec = np.real(eigvec)[:, idx0[0]].T
        w = eigvec[:, :-number_of_constraints]
        w0 = initial_condition[:, :-number_of_constraints]
        omega = 1/np.sqrt(eigval)
        alpha = np.array([(w[i, :] @ M @ w0.T) / (w[i, :] @ M @ w[i, :]) for i in range(omega.shape[1])]).T
        u = np.zeros((self.total_time_steps, Me.shape[0]))
        frame_count = 0
        temp = [initial_condition]
        time = [0]
        for i in range(self.total_time_steps):
            u[i, :] = (alpha * np.cos(omega*i*dt) @ eigvec).flatten()
            if i*dt >= frame_count/fps:
                temp.append(u[i, :])
                time.append((i+1)*dt)  
                frame_count += 1
        results = np.zeros((len(temp), Me.shape[0]))
        for i in range(len(temp)):
            results[i, :] = temp[i]
        return time, results

    def get_results(self, solution, grid_size):
        x = np.zeros((solution.shape[0], len(self.beam_list), grid_size))
        y = np.zeros((solution.shape[0], len(self.beam_list), grid_size))
        for i in range(solution.shape[0]):
            start_index = 0
            for j, beam in enumerate(self.beam_list):
                n = beam.mesh_size
                x_beam, y_beam = beam.get_results(solution[i, start_index : start_index + 3*(n+1)], grid_size)
                start_index += 3*(n+1)
                x[i, j, :] = x_beam
                y[i, j, :] = y_beam
        return x, y

    def calculate_maximum_deformation(self, static_solution):
        max_deformation = np.zeros(len(self.beam_list))
        start_index = 0
        for j, beam in enumerate(self.beam_list):
            n = beam.mesh_size
            coeff = static_solution[0, start_index : start_index + 3*(n+1)]
            grid = np.linspace(0, beam.length, 100)
            w = beam.hermitian_interpolator(coeff[n+1:])
            max_deformation[j] = max(abs(w(grid)))
            start_index += 3*(n+1)
        return max_deformation

## Auxiliary functions for creating frameworks 
def create_beam(node_pair, node_list, n, physical_parameters, initial_load_longitudinal = lambda _: 0, initial_load_transversal = lambda _: 0):
    p_start = np.array(node_list[node_pair[0]])
    p_end = np.array(node_list[node_pair[1]])
    d = p_end - p_start
    L = norm(d)
    phi = np.arctan2(d[1], d[0])
    beam = Beam(p_start, phi, L, n, physical_parameters, initial_load_longitudinal = initial_load_longitudinal, initial_load_transversal = initial_load_transversal)
    return beam

def get_cylinder_area(dia=0.05):
    return np.pi*dia**2/4.0