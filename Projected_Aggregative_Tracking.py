#DAS Project 2023/24
#Task 2.3 - Projected Aggregative Tracking Optional Task
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

#agents' parameters
p_ER = 0.3 #probability of edge
NN = 6 #number of agents
n_z = 2 #agents' dimension

#simulation's parameters
MAXITERS = 300 #maximum number of iterations
step_size = 1e-2 #stepsize of the algorithm
animation = True #allow animation
dt = 1 #sub-sampling of the plot horizon
reproducibility = True #simulation's reproducibility
if reproducibility: np.random.seed(0)

#online aggregative optimization's parameters
show_intermediate_targets = True #show intermediate targets
delta = 0.8
alpha = step_size
walls_margin = 1.5 #degree of margin for corridor's projection

#gains of the cost function's terms
gamma_r_lt = None #robot wrt the local target
gamma_agg = None #barycenter's attractive force
gamma_bar = None #barycenter's speed
gamma_rep = None #barycenter's repulsive force

#corridor's parameters
corr_length = 19 #length of the corridor
corr_width = 6 #width of the corridor
corr_pos = (30,5) #center of corridor's coordinates
corr_bound_offset = 6 #offsets for corridor's entrance and exit
show_corr_points = False

#targets' parameters
target_avg = (60,5) #central target position
initial_avg = (0,0) #starting point of the robots' formation
radius = 9.0 #radius of the circle
gain = 0.8 #gain controlling the degree of closeness
b = np.zeros(n_z) #barycenter target

#start the programs
task = "2.3 Projected_Aggregative_Tracking"
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

#initialize local targets of the robots
RR = np.zeros((NN,n_z))

#arrays to store cost function and gradient norms' values
cost_values = np.zeros(MAXITERS)

#arrays to store cost function and gradient norms' values
grad_z = np.zeros((MAXITERS, n_z))
grad_s = np.zeros((MAXITERS, n_z))
true_barycenter = np.zeros((MAXITERS, n_z))

#find the corridor's points
corr_points = funcs.generate_corridor_points(corr_length, corr_width, corr_pos, "proj")

#define the corridor's entrance and exit
corr_entrance = (corr_pos[0] - corr_length / 2 - corr_bound_offset, corr_pos[1])
corr_exit = (corr_pos[0] + corr_length / 2 + corr_bound_offset, corr_pos[1])

#define the optimization problems
optimization_problems = {

    "to_corridor": { #from starting point to the beginning of the corridor

        "barycenter_goal_coordinates" : corr_entrance, #initial point of the corridor    
        "local_targets_coordinates" : funcs.generate_circular_positions(NN, n_z, corr_entrance, radius, gain),
        "barycenter_attraction_gain": 1,
        "barycenter_repulsion_gain": 0,
        "barycenter_speed_gain": 3,
        "local_targets_attraction_gain": 0.8,
        "walls_obstacle_avoidance": False,
    },

    "through_corridor": { #from the beginning to the end of the corridor

        "barycenter_goal_coordinates" : corr_exit, #initial point of the corridor    
        "local_targets_coordinates" : funcs.generate_circular_positions(NN, n_z, corr_exit, radius, gain),
        "barycenter_attraction_gain": 1,
        "barycenter_repulsion_gain": 0, #adjust the repulsion degree of the barycenter
        "barycenter_speed_gain": 3, #adjust the speed of the formation
        "local_targets_attraction_gain": 0.8,
        "walls_obstacle_avoidance": True,
    },

    "to_targets": { #from the end of the corridor to the final targets

        "barycenter_goal_coordinates" : target_avg,
        "local_targets_coordinates" : funcs.generate_circular_positions(NN, n_z, target_avg, radius, gain),
        "barycenter_attraction_gain": 1, #degree of attraction of the barycenter
        "barycenter_repulsion_gain": 0,
        "barycenter_speed_gain": 2,
        "local_targets_attraction_gain": 4,
        "walls_obstacle_avoidance": False,
    },
}

#initialize the overall trajectory
overall_ZZ = np.zeros((NN, MAXITERS, n_z))
overall_SS = np.zeros((NN, MAXITERS, n_z))
overall_VV = np.zeros((NN, MAXITERS, n_z))
overall_FF = np.zeros((MAXITERS))

#initialize the gradients' variables
overall_grad_z = np.zeros((MAXITERS, n_z))
overall_grad_s = np.zeros((MAXITERS, n_z))
grad_norm_z = np.zeros((MAXITERS))
grad_norm_s = np.zeros((MAXITERS))

# Calculate the number of iterations each optimization problem should run for
iters_per_problem = MAXITERS // len(optimization_problems)
overall_iter_index = 0
print("Iterations per problem are: ", iters_per_problem)

# Initialize the initial state for the first optimization problem
ZZ_init = None

#list to store the local targets
local_targets = []

#############################################################################
############################ Aggregative Algorithm ##########################
#############################################################################

for _, problem_params in optimization_problems.items():

    #extract the behaviors' parameters from the current optimization problem
    b[:2] = problem_params["barycenter_goal_coordinates"]
    RR = problem_params["local_targets_coordinates"]
    gamma_agg = problem_params["barycenter_attraction_gain"]
    gamma_rep = problem_params["barycenter_repulsion_gain"]
    gamma_bar = problem_params["barycenter_speed_gain"]
    gamma_r_lt = problem_params["local_targets_attraction_gain"]
    avoid_walls = problem_params["walls_obstacle_avoidance"]

    #save the local tergets for plotting purposes
    local_targets.append(RR)

    #initialize the trajectory for the current problem
    ZZ = np.zeros((NN, iters_per_problem, n_z))
    SS = np.zeros((NN, iters_per_problem, n_z))
    VV = np.zeros((NN, iters_per_problem, n_z))
    FF = np.zeros((iters_per_problem))

    #initialize the gradients for the current problem
    grad_z = np.zeros((iters_per_problem, n_z))
    grad_s = np.zeros((iters_per_problem, n_z))

    #initialize the robots randomly in a radial way
    if ZZ_init is None:
        
        #initialize the robots in a radial way
        ZZ_init = funcs.generate_circular_positions(NN, n_z, initial_avg, radius, gain)

        #set robots' initial positions
        ZZ[:, 0, :] = ZZ_init

    #use the final state of the previous problem as the initial state
    else:
        ZZ[:, 0, :] = ZZ_init
        print("Setting initial state: ", ZZ_init)

    #launch the online algorithm to track moving local targets
    print("Projected Aggregative Tracking.")
    ZZ_current, SS_current, FF_current, VV_current, WW, RR, grad_s_current, grad_z_current = funcs.projected_aggregative_tracking(NN, iters_per_problem, step_size, alpha, delta, n_z, Adj, ZZ, SS, FF, VV, WW, RR, grad_z, grad_s, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep=gamma_rep, walls_avoidance=avoid_walls, b=b, corr_points=corr_points, walls_margin=walls_margin)

    #update the overall trajectory with the trajectory of the current problem
    overall_ZZ[:, overall_iter_index:overall_iter_index + iters_per_problem, :] = ZZ_current
    overall_SS[:, overall_iter_index:overall_iter_index + iters_per_problem, :] = SS_current
    overall_VV[:, overall_iter_index:overall_iter_index + iters_per_problem, :] = VV_current
    overall_FF[overall_iter_index:overall_iter_index + iters_per_problem] = FF_current
    overall_grad_s[overall_iter_index:overall_iter_index + iters_per_problem, :] = grad_z_current
    overall_grad_z[overall_iter_index:overall_iter_index + iters_per_problem, :] = grad_s_current
    
    #update the overall iteration index
    overall_iter_index += iters_per_problem

    #update the initial state for the next optimization problem
    ZZ_init = ZZ_current[:, -1, :]

#############################################################################
################################## Plots ####################################
#############################################################################

##############################
############ Cost ############
##############################

#plot the cost function
plt.figure()
#plt.plot(np.arange(MAXITERS), np.abs(overall_FF-overall_FF[-1]), '--', linewidth=3)
plt.plot(np.arange(MAXITERS), overall_FF, '--', linewidth=3)
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
grad_norm_z = np.linalg.norm(overall_grad_z, axis=1)  
grad_norm_s = np.linalg.norm(overall_grad_s, axis=1)

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

#############################################################################
############################### Animation ###################################
#############################################################################

'''If enabled and the dimensionality is 2, this part of the code plots an animation
showing the trajectory of agents, initial positions, target position, and estimate
of the centroid, i.e., the barycenter of the team.'''

#animation function
if animation and n_z == 2: funcs.robots_animation(NN, MAXITERS, overall_ZZ, overall_SS, RR, b, n_z, dt, corr_length, corr_width, corr_pos, local_targets, 'static', show_intermediate_targets, corr_points=corr_points, show_corr_points=show_corr_points)

#end the program
funcs.message_handler("end", task)
