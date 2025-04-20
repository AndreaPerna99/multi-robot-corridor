#DAS Project 2023/24
#Task 2.3
#Gianluca Di Mauro, Andrea Perna, Meisam Tavakoli

#############################################################################
############################### Libraries ###################################
#############################################################################

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import Project_Functions as funcs

#############################################################################
############################### Parameters ##################################
#############################################################################

#simulation's parameters
MAXITERS = 1000 #maximum number of iterations
step_size = 1e-2 #stepsize of the algorithm
animation = True #allow animation
dt = 10 #sub-sampling of the plot horizon
reproducibility = True #simulation's reproducibility
if reproducibility: np.random.seed(0)

#gains of the cost function's terms
gamma_r_lt = 0.9 #robot wrt the local target
gamma_agg = 0 #barycenter's attractive force
gamma_bar = 0.2 #barycenter's speed
gamma_rep = 0 #barycenter's repulsive force
avoid_walls = True

#corridor's parameters
corr_length = 19 #length of the corridor
corr_width = 10 #width of the corridor
corr_pos = (30,5) #center of corridor's coordinates
corr_bound_offset = 6 #offsets for corridor's entrance and exit
show_corr_points = True

#local targets' parameters
target_avg = (60,5) #central target position
initial_avg = (0,5) #starting point of the robots' formation
radius = 6 #radius of the circle
gain = 0.7 #gain controlling the degree of closeness
show_intermediate_targets=False

#agents' parameters
p_ER = 0.3 #probability of edge
NN = 6 #number of agents
n_z = 2 #agents' dimension
b = np.zeros(n_z) #barycenter target
b = target_avg
random_init = False

#potential function's parameters
qstar = 5
K = 1000

#start the programs
task = 2.3
funcs.message_handler("title", task)

#############################################################################
############################ Graph Generation ###############################
#############################################################################

'''This part of the code generates a connected graph with a given probability
parameter p_ER. It also ensures that the generated graph is connected.
Also, it constructs an adjacency matrix based on the generated graph, by
assigning weights to edges based on the degree of connectivity between nodes.'''

#create identity matrix
I_NN = np.identity(NN, dtype=int)

#generate a connected graph
Adj = funcs.generate_connected_graph(NN, I_NN, p_ER)

#assign the Metropolis-Hastings weights to the graph's edges
WW = funcs.metropolis_hastings(NN, Adj, I_NN)

#############################################################################
############################# Initialization ################################
#############################################################################

'''this initialization prepares the necessary data structures to execute
the optimization algorithm and track its progress, including cost function
evaluations, gradient norms, and robot positions.'''

#define the corridor
corr_points = funcs.generate_corridor_points(corr_length, corr_width, corr_pos, "pot")
print("Corr_points: ", corr_points)

#initialize local targets of the robots
RR = np.zeros((NN,n_z))
RR = funcs.generate_circular_positions(NN, n_z, target_avg, radius, gain)
local_targets = []

#save the local tergets for plotting purposes
local_targets.append(RR)

#initialize the trajectory for the current problem
ZZ = np.zeros((NN, MAXITERS, n_z))
SS = np.zeros((NN, MAXITERS, n_z))
VV = np.zeros((NN, MAXITERS, n_z))

#initialize the cost and the gradients
FF = np.zeros((MAXITERS))
grad_z = np.zeros((MAXITERS, n_z))
grad_s = np.zeros((MAXITERS, n_z))
grad_norm_z = np.zeros((MAXITERS))
grad_norm_s = np.zeros((MAXITERS))
true_barycenter = np.zeros((MAXITERS, n_z))

#initialize the robots in a radial way
if not random_init:  ZZ_init = funcs.generate_circular_positions(NN, n_z, initial_avg, radius, gain)

#initialize the robots randomly
else: ZZ_init = funcs.random_initialization(NN, n_z, corr_points)

#set robots' initial positions
ZZ[:, 0, :] = ZZ_init

#############################################################################
############################ Aggregative Algorithm ##########################
#############################################################################

#launch the aggregative gradient algorithm
print("Aggregative Gradient Tracking.")
ZZ, SS, FF, VV, WW, RR, grad_s, grad_z, true_barycenter = funcs.aggregative_gradient(NN, MAXITERS, step_size, n_z, Adj, ZZ, SS, FF, VV, WW, RR, grad_z, grad_s, gamma_r_lt, gamma_bar, gamma_agg, b=b, local_targets=local_targets, corr_points=corr_points, avoid_walls=avoid_walls, true_barycenter=true_barycenter, qstar=qstar, K=K)

#############################################################################
################################## Plots ####################################
#############################################################################

##############################
############ Cost ############
##############################

#plot the cost function
plt.figure()
#plt.plot(np.arange(MAXITERS), np.abs(FF-FF[-1]), '--', linewidth=3)
plt.plot(np.arange(MAXITERS), FF, '--', linewidth=3)
plt.yscale('log')        
plt.xlabel(r"Iterations $t$")
plt.ylabel(r"Cost")
plt.title("Evolution of the cost function")
plt.grid()
plt.show()

##############################
####### Gradient Norms #######
##############################

#compute the gradient norms   
grad_norm_z = np.linalg.norm(grad_z, axis=1)  
grad_norm_s = np.linalg.norm(grad_s, axis=1)

# Create a figure with subplots arranged horizontally
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot the gradient norm of agent's positions
ax[0].plot(np.arange(MAXITERS), grad_norm_z)
ax[0].set_yscale('log')
ax[0].set_xlabel(r"Iterations $t$")
ax[0].set_ylabel(r"ZZ Gradient Norm")
ax[0].set_title("Evolution of the agents' position gradient norm")
ax[0].grid()

# Plot the gradient norm of team's barycenter
ax[1].plot(np.arange(MAXITERS), grad_norm_s)
ax[1].set_yscale('log')
ax[1].set_xlabel(r"Iterations $t$")
ax[1].set_ylabel(r"SS Gradient Norm")
ax[1].set_title("Evolution of the team's barycenter gradient norm")
ax[1].grid()

# Adjust layout for better readability
plt.tight_layout()
plt.show()

##############################
####### Tracking Errors ######
##############################

# Create a figure with subplots arranged horizontally
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot the gradient norm of agent's positions
for ii in range(NN): ax[0].plot(np.arange(MAXITERS), np.linalg.norm(SS[ii, :, :] - true_barycenter, axis=1))
ax[0].set_yscale('log')
ax[0].set_xlabel(r"Iterations $t$")
ax[0].set_ylabel(r"SS - true_barycenter")
ax[0].set_title("Convergence of barycenter's tracker")
ax[0].grid()

# Plot the gradient norm of team's barycenter
for ii in range(NN): ax[1].plot(np.arange(MAXITERS), np.linalg.norm(VV[ii, :, :] - grad_s, axis=1))
ax[1].set_yscale('log')
ax[1].set_xlabel(r"Iterations $t$")
ax[1].set_ylabel(r"VV - grad_s")
ax[1].set_title("Convergence of grad_s's tracker")
ax[1].grid()

# Adjust layout for better readability
plt.tight_layout()
plt.show()

#############################################################################
############################### Animation ###################################
#############################################################################

'''If enabled and the dimensionality is 2, this part of the code plots an animation
showing the trajectory of agents, initial positions, target position, and estimate
of the centroid, i.e., the barycenter of the team.'''

#animation function
if animation and n_z == 2: funcs.robots_animation(NN, MAXITERS, ZZ, SS, RR, b, n_z, dt, corr_length, corr_width, corr_pos, local_targets, 'static', show_intermediate_targets, corr_points=corr_points, show_corr_points=show_corr_points)

#end the program
funcs.message_handler("end", task)
