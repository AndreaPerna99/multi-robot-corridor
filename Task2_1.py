#Task 2.1
#Gianluca Di Mauro, Andrea Perna, Meisam Tavakoli

#############################################################################
############################### Libraries ###################################
#############################################################################

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import ast
import Project_Functions as funcs
import networkx as nx

#############################################################################
############################### Parameters ##################################
#############################################################################

#agents' parameters
NN = 5 #number of agents
n_z = 2 #agents' dimension

#simulation's parameters
MAXITERS = 700 #maximum number of iterations
step_size = 1e-2 # stepsize of the algorithm
animation = True #allow animation
dt = 3 # sub-sampling of the plot horizon
reproducibility = False #simulation's reproducibility
if reproducibility: np.random.seed(0)

#online aggregative optimization's parameters
targets_motion_type = 'hor' #choose either hor, vert or diag
moving_loc_targets = False #allow local targets' motion
show_intermediate_targets = True #show intermediate targets
targets_shift_x = 1
targets_shift_y = 1
targets_rate_change = 100 #number of iterations to skip
alpha = step_size

#local targets' parameters
target_avg = (2,2) #central target position
initial_avg = (0,0) #starting point of the robots' formation
radius = 3.0 #radius of the circle
gain = 0.5 #gain controlling the degree of closeness
random_init = False #generate randiìom initial points

#global target
b = np.array([2,2]) #barycenter target

#graph parameters
p_ER = 0.3 #probability of edge

#gains of the cost function's terms
gamma_r_lt = 0.9 #robot wrt the local target
gamma_agg = 0 #robot wrt team's barycenter
gamma_bar = 0.2 #barycenter wrt formation's target

#start the program
task = 2.1
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

#initialize cost and decision vector
FF = np.zeros((MAXITERS))
ZZ = np.zeros((NN,MAXITERS,n_z))

#initialize the robots randomly in a radial way
ZZ_init = funcs.generate_circular_positions(NN, n_z, initial_avg, radius, gain)
ZZ[:, 0, :] = ZZ_init

#initialize local targets of the robots 
RR = np.zeros((NN,n_z))
RR = funcs.generate_circular_positions(NN, n_z, target_avg, radius, gain)

#initialize the global terms
SS = np.zeros((NN,MAXITERS,n_z)) #barycenter
VV = np.zeros((NN,MAXITERS,n_z)) #gradient

#arrays to store cost function and gradient norms' values
grad_z = np.zeros((MAXITERS, n_z))
grad_s = np.zeros((MAXITERS, n_z))
grad_norm_z = np.zeros((MAXITERS))
grad_norm_s = np.zeros((MAXITERS))
true_barycenter = np.zeros((MAXITERS, n_z))

#list to store the local targets
local_targets = []
local_targets.append(RR)

#############################################################################
############################ Aggregative Algorithm ##########################
#############################################################################

'''This part of the code implements the distributed aggregative optimization
algorithm. It iterates through each agent and updates its decision variable,
estimate of the aggregative variable, and sum of gradients based on the cost
function and neighboring agents' information.'''
    
#launch the aggregative gradient algorithm
print("Aggregative Gradient Tracking.")
ZZ, SS, FF, VV, WW, RR, grad_s, grad_z, true_barycenter = funcs.aggregative_gradient(NN, MAXITERS, step_size, n_z, Adj, ZZ, SS, FF, VV, WW, RR, grad_z, grad_s, gamma_r_lt, gamma_bar, gamma_agg, b=b, local_targets=local_targets, targets_motion_type=targets_motion_type, shift_x=targets_shift_x, shift_y=targets_shift_y, targets_rate_change=targets_rate_change, moving_targets=moving_loc_targets, true_barycenter=true_barycenter)

#############################################################################
################################### Plots ###################################
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

# animation
if animation and n_z == 2: funcs.robots_animation(NN, MAXITERS, ZZ, SS, RR, b, n_z, dt, show_intermediate_targets=show_intermediate_targets, local_targets=local_targets)

#end the program
funcs.message_handler("end", task)
