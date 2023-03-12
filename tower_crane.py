import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow
from framework import Framework, create_beam, get_cylinder_area

# structure steel
physical_parameters = dict()
physical_parameters['modulus'] = 2e11
physical_parameters['density'] = 7.85e3
physical_parameters['area'] = get_cylinder_area(dia=0.3)

physical_parameters_2 = dict()
physical_parameters_2['modulus'] = 2e11
physical_parameters_2['density'] = 7.85e3
physical_parameters_2['area'] = 0.1

n = 10

vertical_blocks = 16
horizontal_left_blocks = 4
connection_block_horizontal_vertical = 14
horizontal_right_blocks = 14
first_cable_block = 4
second_cable_block = 9
weight_left_block = 4
weight_right_block = 9
cable_length = 8
width_weight = 1
height_weight = 2

weight_left = 1e7
weight_right = 5e7

q = 0 # for plotting the framework setup and element distribution
#q = -9e4

# initialization
nodes = []
beams = []
Construction = Framework()

# function for generating the repeating |__| plus diagonal / 4-beam sub-structure

start_beam_vertical_structure = 1
start_node_vertical_structure = 0
# generate the vertical structure
for i in range(vertical_blocks):
    # node index
    k = start_node_vertical_structure + 2*i
    # the first repeating structure
    if i == 0:
        nodes.append(np.array([0,0]))
        nodes.append(np.array([1,0]))
        nodes.append(np.array([0,1]))
        nodes.append(np.array([1,1]))
        beams.append((0, 1))
        beams.append((0, 2))
        beams.append((0, 3))
        beams.append((1, 3))
    # the last repeating structure
    elif i == (vertical_blocks-1):
        node_k = nodes[k]
        nodes.append(node_k + np.array([1, 1]))
        beams.append((k, k+1))
        beams.append((k, k+2))
        beams.append((k+1, k+2))
    else:
        node_k = nodes[k]
        nodes.append(node_k + np.array([0, 1]))
        nodes.append(node_k + np.array([1, 1]))
        beams.append((k, k+1))
        beams.append((k, k+2))
        beams.append((k, k+3))
        beams.append((k+1, k+3))

# generate the horizontal structures on the left and the right
start_beam_horizontal_left_structure = len(beams) + 1
start_node_horizontal_left_structure = len(nodes)
for i in range(horizontal_left_blocks):
    k = 2*(max(0, i-1)) + start_node_horizontal_left_structure
    if i == 0:
        nodes.append(np.array([-1, connection_block_horizontal_vertical-1]))
        nodes.append(np.array([-1, connection_block_horizontal_vertical]))
        beams.append((2*(connection_block_horizontal_vertical-1), k))
        beams.append((2*connection_block_horizontal_vertical, k))
        beams.append((2*connection_block_horizontal_vertical, k+1))
    else:
        node_k = nodes[k]
        nodes.append(node_k + np.array([-1, 0]))
        nodes.append(node_k + np.array([-1, 1]))
        beams.append((k, k+1))
        beams.append((k, k+2))
        beams.append((k+1, k+2))
        beams.append((k+1, k+3))
    if i == (horizontal_left_blocks - 1):
        if i == 0:
            beams.append((k, k+1))
        else:
            beams.append((k+2, k+3))

start_beam_horizontal_right_structure = len(beams) + 1
start_node_horizontal_right_structure = len(nodes)
for i in range(horizontal_right_blocks):
    k = 2*(max(0, i-1)) + start_node_horizontal_right_structure
    if i == 0:
        nodes.append(np.array([2, connection_block_horizontal_vertical]))
        nodes.append(np.array([2, connection_block_horizontal_vertical-1]))
        beams.append((2*connection_block_horizontal_vertical+1, k))
        beams.append((2*(connection_block_horizontal_vertical-1)+1, k))
        beams.append((2*(connection_block_horizontal_vertical-1)+1, k+1))
    else:
        node_k = nodes[k]
        nodes.append(node_k + np.array([1, 0]))
        nodes.append(node_k + np.array([1, -1]))
        beams.append((k, k+1))
        beams.append((k, k+2))
        beams.append((k+1, k+2))
        beams.append((k+1, k+3))
    if i == (horizontal_right_blocks - 1):
        if i == 0:
            beams.append((k, k+1))
        else:
            beams.append((k+2, k+3))

# connect the top to the horizontal strctures
start_beam_cables = len(beams) + 1

beams.append((start_node_horizontal_left_structure-1, start_node_horizontal_right_structure-1))
beams.append((start_node_horizontal_left_structure-1, start_node_horizontal_right_structure + 2*(first_cable_block-2)))
beams.append((start_node_horizontal_left_structure-1, start_node_horizontal_right_structure + 2*(second_cable_block-2)))

# connect the weight to the horizontal structures
start_beam_weight = len(beams) + 1
start_node_weight = len(nodes)

connection_point = np.array([weight_right_block, connection_block_horizontal_vertical - cable_length - 1])
nodes.append(connection_point)
nodes.append(connection_point + [-width_weight/2, 0])
nodes.append(connection_point + [width_weight/2, 0])
nodes.append(connection_point + [-width_weight/2, -height_weight])
nodes.append(connection_point + [width_weight/2, -height_weight])

beams.append((start_node_horizontal_right_structure + 2*(weight_right_block-2)+1, start_node_weight))
beams.append((start_node_weight, start_node_weight + 1))
beams.append((start_node_weight, start_node_weight + 2))
beams.append((start_node_weight+1, start_node_weight+3))
beams.append((start_node_weight+1, start_node_weight+4))
beams.append((start_node_weight+2, start_node_weight+3))
beams.append((start_node_weight+2, start_node_weight+4))
beams.append((start_node_weight+3, start_node_weight+4))

# add all beams to the framework
for i in range(len(beams)):
    if i == start_beam_weight:
        Construction.add_beam(create_beam(beams[i], nodes, n, physical_parameters, initial_load_longitudinal = weight_right))
    elif i == (start_beam_horizontal_right_structure-1):
        Construction.add_beam(create_beam(beams[i], nodes, n, physical_parameters, initial_load_longitudinal = -weight_left))
    else:
        Construction.add_beam(create_beam(beams[i], nodes, n, physical_parameters))

# add constraints to the vertical structure
Construction.add_constraint("fixed_bearing", [start_beam_vertical_structure + 1], ["left"])
Construction.add_constraint("fixed_bearing", [start_beam_vertical_structure + 3], ["left"])

for i in range(vertical_blocks):
    # node index
    k = start_beam_vertical_structure + 4*i
    if i == (vertical_blocks - 1):
        Construction.add_constraint("linking", [k, k+1], ["left", "left"])
        Construction.add_constraint("linking", [k, k+2], ["right", "left"])
        Construction.add_constraint("linking", [k+1, k+2], ["right", "right"])
        # only three beams in this sub-structrue
        Construction.add_constraint("linking", [k, k-3], ["left", "right"])
        Construction.add_constraint("linking", [k+2, k+2-3], ["left", "right"])
        Construction.add_constraint("stiffness_of_angles", [k+2, k+2-3], ["left", "right"])
    else:
        Construction.add_constraint("linking", [k, k+1], ["left", "left"])
        Construction.add_constraint("linking", [k, k+2], ["left", "left"])
        Construction.add_constraint("linking", [k, k+3], ["right", "left"])    
        Construction.add_constraint("linking", [k+2, k+3], ["right", "right"])
        if i > 0:
            Construction.add_constraint("linking", [k+1, k+1-4], ["left", "right"])
            Construction.add_constraint("stiffness_of_angles", [k+1, k+1-4], ["left", "right"])
            Construction.add_constraint("linking", [k+3, k+3-4], ["left", "right"])
            Construction.add_constraint("stiffness_of_angles", [k+3, k+3-4], ["left", "right"])


# add constraints to the horizontal structures
for i in range(horizontal_left_blocks):
    # node index
    k = 4*i + start_beam_horizontal_left_structure - 1
    # connect horizontal structure to vertical structures
    if i == 0:
        Construction.add_constraint("linking", [2 + 4*(connection_block_horizontal_vertical-1), k+1], ["left", "left"])
        Construction.add_constraint("linking", [2 + 4*(connection_block_horizontal_vertical-1), k+2], ["right", "left"])
        Construction.add_constraint("linking", [2 + 4*(connection_block_horizontal_vertical-1), k+3], ["right", "left"])    
        Construction.add_constraint("linking", [k+1, k+2], ["right", "right"])
        Construction.add_constraint("stiffness_of_angles", [k+1, 1+4*(connection_block_horizontal_vertical-1)], ["left", "left"])
        Construction.add_constraint("stiffness_of_angles", [k+3, 5+4*(connection_block_horizontal_vertical-1)], ["left", "left"])
    # connect the main neighbouring blocks
    else:
        Construction.add_constraint("linking", [k, k+1], ["left", "left"])
        Construction.add_constraint("linking", [k, k+2], ["right", "left"])
        Construction.add_constraint("linking", [k, k+3], ["right", "left"])    
        Construction.add_constraint("linking", [k+1, k+2], ["right", "right"])

        Construction.add_constraint("linking", [k+1, k+1-4], ["left", "right"])
        Construction.add_constraint("stiffness_of_angles", [k+1, k+1-4], ["left", "right"])
        Construction.add_constraint("linking", [k+3, k+3-4], ["left", "right"])
        Construction.add_constraint("stiffness_of_angles", [k+3, k+3-4], ["left", "right"])
    if i == (horizontal_left_blocks - 1):
        Construction.add_constraint("linking", [k+1, k+4], ["right", "left"])
        Construction.add_constraint("linking", [k+3, k+4], ["right", "right"])

for i in range(horizontal_right_blocks):
    # node index
    k = 4*i + start_beam_horizontal_right_structure - 1
    # connect horizontal structure to vertical structures
    if i == 0:
        Construction.add_constraint("linking", [4 + 4*(connection_block_horizontal_vertical-1), k+1], ["right", "left"])
        Construction.add_constraint("linking", [4 + 4*(connection_block_horizontal_vertical-1), k+2], ["left", "left"])
        Construction.add_constraint("linking", [4 + 4*(connection_block_horizontal_vertical-1), k+3], ["left", "left"])    
        Construction.add_constraint("linking", [k+1, k+2], ["right", "right"])
        Construction.add_constraint("stiffness_of_angles", [k+1, 5+4*(connection_block_horizontal_vertical-1)], ["left", "right"])
        Construction.add_constraint("stiffness_of_angles", [k+3, 1+4*(connection_block_horizontal_vertical-1)], ["left", "right"])
    # connect the main neighbouring blocks
    else:
        Construction.add_constraint("linking", [k, k+1], ["left", "left"])
        Construction.add_constraint("linking", [k, k+2], ["right", "left"])
        Construction.add_constraint("linking", [k, k+3], ["right", "left"])    
        Construction.add_constraint("linking", [k+1, k+2], ["right", "right"])

        Construction.add_constraint("linking", [k+1, k+1-4], ["left", "right"])
        Construction.add_constraint("stiffness_of_angles", [k+1, k+1-4], ["left", "right"])
        Construction.add_constraint("linking", [k+3, k+3-4], ["left", "right"])
        Construction.add_constraint("stiffness_of_angles", [k+3, k+3-4], ["left", "right"])
    if i == (horizontal_right_blocks - 1):
        Construction.add_constraint("linking", [k+1, k+4], ["right", "left"])
        Construction.add_constraint("linking", [k+3, k+4], ["right", "right"])   

# add constraints between cables and horizontal structures
Construction.add_constraint("linking", [start_beam_cables, start_beam_horizontal_left_structure-1], ["left", "right"])
Construction.add_constraint("linking", [start_beam_cables, start_beam_horizontal_right_structure-1], ["right", "right"])

Construction.add_constraint("linking", [start_beam_cables+1, start_beam_horizontal_left_structure-1], ["left", "right"])
Construction.add_constraint("linking", [start_beam_cables+1, 4*(first_cable_block-1) + start_beam_horizontal_right_structure-1], ["right", "left"])

Construction.add_constraint("linking", [start_beam_cables+2, start_beam_horizontal_left_structure-1], ["left", "right"])
Construction.add_constraint("linking", [start_beam_cables+2, 4*(second_cable_block-1) + start_beam_horizontal_right_structure-1], ["right", "left"])

# add constraints between the weight and the horizontal structure
Construction.add_constraint("linking", [start_beam_weight, 4*(weight_right_block-1) + start_beam_horizontal_right_structure-1], ["left", "right"])
Construction.add_constraint("stiffness_of_angles", [start_beam_weight, 4*(weight_right_block-1) + start_beam_horizontal_right_structure-1], ["left", "right"])
Construction.add_constraint("linking", [start_beam_weight, start_beam_weight+1], ["right", "left"])
Construction.add_constraint("stiffness_of_angles", [start_beam_weight, start_beam_weight+1], ["right", "left"])
Construction.add_constraint("linking", [start_beam_weight+1, start_beam_weight+2], ["left", "left"])
Construction.add_constraint("stiffness_of_angles", [start_beam_weight+1, start_beam_weight+2], ["left", "left"])

Construction.add_constraint("linking", [start_beam_weight+1, start_beam_weight+3], ["right", "left"])
Construction.add_constraint("linking", [start_beam_weight+1, start_beam_weight+4], ["right", "left"])
Construction.add_constraint("linking", [start_beam_weight+2, start_beam_weight+5], ["right", "left"])
Construction.add_constraint("linking", [start_beam_weight+2, start_beam_weight+6], ["right", "left"])
Construction.add_constraint("linking", [start_beam_weight+7, start_beam_weight+3], ["left", "right"])
Construction.add_constraint("linking", [start_beam_weight+7, start_beam_weight+4], ["right", "right"])
Construction.add_constraint("linking", [start_beam_weight+7, start_beam_weight+5], ["left", "right"])
Construction.add_constraint("linking", [start_beam_weight+7, start_beam_weight+6], ["right", "right"])

# dynamic solution sovler
T = 1.6
dt = 1e-4
fps = 30
number_of_eig = 10

kappa = cond(Construction.get_extended_stiffness_matrix().toarray())
if kappa > 1e16:
    raise Exception("System is ill-conditioned! condition number = {cond}".format(cond = kappa))
else:
    print("Condition number = {:e}".format(kappa))

initial_condition = Construction.static_solve()
max_deformation = Construction.calculate_maximum_deformation(initial_condition)
# print(max_deformation)
print(max(max_deformation))

"""
# use arrows to indicate loading for simple crane

l,_ = p7 - p6
arrow_x, arrow_y = p6
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
x_limits = (-5, 16)
y_limits = (-1, 17)
aspect_ratio = (x_limits[1] - x_limits[0])/(y_limits[1] - y_limits[0])
width = 18
fig = plt.figure(figsize = (width, width/aspect_ratio))
ax = plt.axes(xlim = x_limits, ylim = y_limits)
plt.show()

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
anim.save('tower_crane.gif', writer = 'imagemagick', fps = fps)
plt.show()


