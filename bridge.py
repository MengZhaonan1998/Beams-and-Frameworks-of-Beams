import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond
from matplotlib.animation import FuncAnimation
from framework import Framework, create_beam

Construction = Framework()
nodes = []
beams = []

nodes.append([0, 0])
nodes.append([2, 0])
nodes.append([1, 0.5*np.sqrt(3)])
nodes.append([4, 0])
nodes.append([3, 0.5*np.sqrt(3)])
nodes.append([6, 0])
nodes.append([5, 0.5*np.sqrt(3)])
nodes.append([8, 0])
nodes.append([7, 0.5*np.sqrt(3)])

n = 10

beams.append((0, 1))
beams.append((1, 2))
beams.append((2, 0))
beams.append((1, 3))
beams.append((4, 1))
beams.append((3, 4))
beams.append((2, 4))
beams.append((3, 5))
beams.append((3, 6))
beams.append((5, 6))
beams.append((4, 6))
beams.append((5, 7))
beams.append((5, 8))
beams.append((7, 8))
beams.append((6, 8))

# Steel beams
physical_parameters = dict()
physical_parameters['modulus'] = 2e11
physical_parameters['density'] = 7.8e3
physical_parameters['area'] = 0.25

q = -1e7
for i in range(len(beams)):
    if i in [0, 3, 7, 11]:
        Construction.add_beam(create_beam(beams[i], nodes, n, physical_parameters, initial_load_transversal = q))
    else:
        Construction.add_beam(create_beam(beams[i], nodes, n, physical_parameters))

Construction.add_constraint("fixed_bearing", [1], ["left"])
Construction.add_constraint("fixed_bearing", [12], ["right"])

Construction.add_constraint("linking", [1, 3], ["left", "right"])
Construction.add_constraint("linking", [1, 2], ["right", "left"])
Construction.add_constraint("linking", [2, 3], ["right", "left"])

Construction.add_constraint("linking", [2, 5], ["left", "right"])
Construction.add_constraint("linking", [5, 7], ["left", "right"])
Construction.add_constraint("linking", [2, 7], ["right", "left"])

Construction.add_constraint("linking", [4, 5], ["left", "right"])
Construction.add_constraint("linking", [5, 6], ["left", "right"])
Construction.add_constraint("linking", [4, 6], ["right", "left"])

Construction.add_constraint("linking", [6, 9], ["left", "left"])
Construction.add_constraint("linking", [9, 11], ["right", "right"])
Construction.add_constraint("linking", [6, 11], ["right", "left"])

Construction.add_constraint("linking", [8, 9], ["left", "left"])
Construction.add_constraint("linking", [9, 10], ["right", "right"])
Construction.add_constraint("linking", [8, 10], ["right", "left"])

Construction.add_constraint("linking", [10, 13], ["left", "left"])
Construction.add_constraint("linking", [13, 15], ["right", "right"])
Construction.add_constraint("linking", [10, 15], ["right", "left"])

Construction.add_constraint("linking", [12, 13], ["left", "left"])
Construction.add_constraint("linking", [13, 14], ["right", "right"])
Construction.add_constraint("linking", [12, 14], ["right", "left"])

Construction.add_constraint("stiffness_of_angles", [1, 4], ["right", "left"])
Construction.add_constraint("stiffness_of_angles", [4, 8], ["right", "left"])
Construction.add_constraint("stiffness_of_angles", [8, 12], ["right", "left"])
Construction.add_constraint("stiffness_of_angles", [7, 11], ["right", "left"])
Construction.add_constraint("stiffness_of_angles", [11, 15], ["right", "left"])

T = 10
dt = 1e-4
fps = 30
number_of_eig = 10

kappa = cond(Construction.get_extended_stiffness_matrix().toarray())
if kappa > 1e16:
    print("System is ill-conditioned! condition number = {cond}".format(cond = kappa))
else:
    print("Condition number = {:e}".format(kappa))

initial_condition = Construction.static_solve()
max_deformation = Construction.calculate_maximum_deformation(initial_condition)
time, solution1 = Construction.dynamic_solve(initial_condition, dt, T, fps)
_, solution2 = Construction.eigen_solve(initial_condition, number_of_eig, dt, T, fps)
x1, y1 = Construction.get_results(solution1, 10)
x2, y2 = Construction.get_results(solution2, 10)
x_limits = (-0.2, 8.2)
y_limits = (-0.2, 1.1)
aspect_ratio = (x_limits[1] - x_limits[0])/(y_limits[1] - y_limits[0])
width = 18
fig = plt.figure(figsize = (width, width/aspect_ratio))
ax = plt.axes(xlim = x_limits, ylim = y_limits)
lines1 = []
lines2 = []
for beam in range(len(Construction.beam_list)):
    line1, = ax.plot([], [], lw = 3, color = 'k')
    line2, = ax.plot([], [], lw = 3, color = 'r')
    lines1.append(line1)
    lines2.append(line2)
    
def init():
    lines1[0].set_label("Newmark")
    lines2[0].set_label("Eigen")
    return 

def animate(i):
    for j, line in enumerate(lines1):
        line.set_data(x1[i, j, :], y1[i, j, :])
    for j, line in enumerate(lines2):
        line.set_data(x2[i, j, :], y2[i, j, :])
    plt.title("time = {time:2.3f} s".format(time = round(time[i], 3)))
    lines1[0].set_label("Newmark")
    lines2[0].set_label("Eigen")
    plt.legend()
    return

print(max_deformation)
print(max(max_deformation))

anim = FuncAnimation(fig, animate, init_func = init, frames = x1.shape[0], interval = 1/fps, repeat_delay = 3000)
anim.save("bridge.gif", writer = 'imagemagick', fps = fps)
plt.show()
