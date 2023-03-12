import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, cond
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow
from framework import Framework, get_cylinder_area, create_beam

Construction = Framework()
nodes = []
beams = []
# the order of node points are dependent of the occurence order of beams, which is in a anti-clockwise manner
# the loading part comes at last
nodes.append([0, 0])
nodes.append([0,np.sqrt(25+3.75**2)])
nodes.append([3, 4])
nodes.append([9, 12])
nodes.append([9, 9])
nodes.append([8, 9])
nodes.append([10, 9])

n = 10
beams.append((0, 1))
beams.append((1, 2))
beams.append((1, 3))
beams.append((4, 3))
beams.append((2, 3))
beams.append((2, 0))
beams.append((4, 5))
beams.append((6, 4))

# structure steel
physical_parameters = dict()
physical_parameters['modulus'] = 2e11
physical_parameters['density'] = 7.85e3
physical_parameters['area'] = get_cylinder_area(dia=0.3)

physical_parameters_2 = dict()
physical_parameters_2['modulus'] = 2e11
physical_parameters_2['density'] = 7.85e3
physical_parameters_2['area'] = 0.1

# q = 0 # for plotting the framework setup and element distribution
q = -1e5

for i in range(len(beams)):
    if i == 6:
        Construction.add_beam(create_beam(beams[i], nodes, n, physical_parameters, initial_load_transversal = q))
    elif i == 7:
        Construction.add_beam(create_beam(beams[i], nodes, n, physical_parameters, initial_load_transversal = q))
    else:
        Construction.add_beam(create_beam(beams[i], nodes, n, physical_parameters))

# External boundary conditions
Construction.add_constraint("fixed_bearing", [6], ["left"])

Construction.add_constraint("linking", [1, 6], ["left", "right"])

# Internal boundary conditions: in the order of triangles that appear from the left to the right
Construction.add_constraint("linking", [1, 2], ["right", "left"])
Construction.add_constraint("linking", [2, 3], ["left", "left"])

Construction.add_constraint("linking", [2, 5], ["right", "left"])
Construction.add_constraint("linking", [2, 6], ["right", "left"])
Construction.add_constraint("stiffness_of_angles", [5, 6], ["left", "left"]) 

Construction.add_constraint("linking", [3, 5], ["right", "right"])
Construction.add_constraint("linking", [4, 5], ["right", "right"])
Construction.add_constraint("stiffness_of_angles", [4, 5], ["right", "right"]) 


Construction.add_constraint("linking", [4, 7], ["left", "left"])
Construction.add_constraint("linking", [4, 8], ["left", "right"])
Construction.add_constraint("stiffness_of_angles", [4, 8], ["left", "right"]) 
Construction.add_constraint("stiffness_of_angles", [7, 8], ["left", "right"]) 

T = 5
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
print(max_deformation)
print(max(max_deformation))

"""
# static solution display
x_static, y_static = Construction.get_results(initial_condition, 10)
fig_static = plt.figure(figsize = (12.8, 3.4))
ax_static = plt.axes(xlim = (-2, 15), ylim = (-2, 16))
for j in range(len(Construction.beam_list)):
    plt.plot(x_static[0, j,:], y_static[0, j,:], lw = 3, color= 'k', marker='o', markersize=5)

# use arrows to indicate loading
l,_ = np.array(nodes[6]) - np.array(nodes[5])
arrow_x, arrow_y = np.array(nodes[5])
arrow_spacing = 0.4
arrow_number = round(l/arrow_spacing) + 1
for i in range(arrow_number):
    arrow = Arrow(arrow_x, arrow_y-0.2, 0, -3, width=0.25)
    ax_static.add_patch(arrow)
    arrow_x += arrow_spacing
plt.show()
"""

time, solution1 = Construction.dynamic_solve(initial_condition, dt, T, fps)
_, solution2 = Construction.eigen_solve(initial_condition, number_of_eig, dt, T, fps)
x1, y1 = Construction.get_results(solution1, 10)
x2, y2 = Construction.get_results(solution2, 10)
x_limits = (-2, 15)
y_limits = (-2, 16)
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

anim = FuncAnimation(fig, animate, init_func = init, frames = x1.shape[0], interval = 1/fps, repeat_delay = 3000)
anim.save('simple_crane.gif', writer = 'imagemagick', fps = fps)
plt.show()

