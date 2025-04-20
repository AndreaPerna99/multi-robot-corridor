#DAS Project 2023/24
#Task 2: Project_Functions.py
#Gianluca Di Mauro, Andrea Perna, Meisam Tavakoli

#############################################################################
############################### Libraries ###################################
#############################################################################

import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import Project_Functions as funcs
import time
import sys
np.random.seed(0)

#############################################################################
################################## Plots ####################################
#############################################################################

def message_handler(mode, task): #hub for messages' handling

    """
    This function prints of the terminal messages related to the program state, according
    to the specific part of the process specified by the input parameter mode.
    """

    #initial plots
    if mode == "title":
        print("\n\n---------------------------------------------------------------")
        print(f"DISTRIBUTED AUTONOMOUS SYSTEMS' PROJECT 2023/24 - TASK {task}")
        print("Andrea Perna")
        print("Gianluca Di Mauro")
        print("Meisam Tavakoli")
        print("All rights reserved")
        print("---------------------------------------------------------------")
        time.sleep(3) #wait few seconds

    #ending plots
    elif mode == "end":

        print("\n\n---------------------------------------------------------------")
        print(f"End of task {task}.")
        print("---------------------------------------------------------------")

def plot_corridor(length, width, corr_pos):

    '''This function generates a corridor of length and width
    specified by the input parameters, with the specified center
    coordinates. Then, corridor is plotted accordingly.'''

    #extract the ceentre of corridor's coordinates
    x_pos = corr_pos[0]
    y_pos = corr_pos[1]
    #calculate half width
    half_width = width / 2

    #generate points for the first upper line
    line1_x = np.array([x_pos - length/2, x_pos + length/2])
    line1_y = np.array([y_pos + half_width, y_pos + half_width])

    #generate points for the second bottom line
    line2_x = np.array([x_pos - length/2, x_pos + length/2])
    line2_y = np.array([y_pos - half_width, y_pos - half_width])

    #save the coordinates
    line1 = (line1_x, line1_y)
    line2 = (line2_x, line2_y)

    #plot the corridor in space
    plt.plot(line1[0], line1[1], 'k-')  # First line
    plt.plot(line2[0], line2[1], 'k-')  # Second line

#############################################################################
############################## Cost Function ################################
#############################################################################

def generate_corridor_points(corr_length, corr_width, corr_pos, mode=None):

    num_points1 = 40

    """Generate points along the edges of the corridor."""
    x_pos, y_pos = corr_pos
    half_width = corr_width / 2
    points = []
    
    # Generate points along both upper and lower edges
    for i in range(num_points1):

        #upper edge point
        x_upper = x_pos - corr_length / 2 + (i / (num_points1 - 1)) * corr_length
        y_upper = y_pos + half_width
        points.append((x_upper, y_upper))
        
        #lower edge point
        x_lower = x_pos - corr_length / 2 + (i / (num_points1 - 1)) * corr_length
        y_lower = y_pos - half_width
        points.append((x_lower, y_lower))

    #extract the entrance points
    upper_left = points[0]  # First point in the list
    lower_left = points[1]  # Second point in the list
    
    print("Upper-left point:", upper_left)
    print("Lower-left point:", lower_left)

    num_points2 = 20

    #generate entrance barrier only for potential function, else proj
    if mode == "pot":

        #generate points over the entrance
        for point in range(num_points2):

            x_p = points[0][0]
            y_ph = points[0][1] + point
            points.append((x_p, y_ph))

        #generate points below the entrance
        for point in range(num_points2):
            x_pl = points[1][0]
            y_pl = points[1][1] - point
            points.append((x_pl, y_pl))

        print("obstacle list points: ", points)
        points = np.array(points)
        print("obstacle array points: ", points)    

    return points

def cost_function(ZZ, SS, RR, b, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep=None, corr_points=None, avoid_walls=None, qstar=None, K=None):
    
    """
    Defines the cost function that needs to be minimized from the distributed aggregative
    optimization algorithm. Based on the boolean input values, it can keeps into account
    the potential function for each point of corr_points, by means of the distances,
    it allows robots to stay close to their local targets while mantaining formation.
    """

    #set zero for None gains
    if gamma_rep is None: gamma_rep = 0

    #number of obstacles
    n_obst = corr_points.shape[0] if corr_points is not None else 0

    #intialize repulsive cost and gradient
    n_z = ZZ.shape[0]
    repulsive_potential = 0
    repulsive_gradient = np.zeros(n_z)
    
    #print("corr_shape: ", len(corr_points))

    if avoid_walls:

        #compute repulsion cost and gradient
        for i in range(n_obst):

            #compute norm of the distance and gradient
            dist = 0.5*(ZZ - corr_points[i]) @ (ZZ - corr_points[i])
            grad_dist = (ZZ - corr_points[i])

            #apply potential function
            if dist <= qstar:
                repulsive_potential += 0.5 * K * ((1 / dist) - (1 / qstar))**2
                repulsive_gradient += K*(((1 / qstar) - (1 / dist)) * grad_dist / dist**2)

    # Define cost sub-functions
    d_robot_target = (ZZ - RR) @ (ZZ - RR)  # Distance between robot (x_i) and its local target (r_i)
    d_barycenter_target = (SS - b) @ (SS - b)  # Distance between estimated barycenter (sigma) and its target (b)
    d_robot_barycenter = (SS - ZZ) @ (SS - ZZ)  # Distance between robot (x_i) and team's estimated barycenter (sigma)

    # Compute distance of each robot from the center
    radial_penalty = (SS - ZZ) @ (SS - ZZ)

    #define the cost function for agent i to be minimized
    f_z = gamma_r_lt * d_robot_target + gamma_bar * d_barycenter_target + gamma_agg * d_robot_barycenter + gamma_rep * radial_penalty + repulsive_potential ###

    #define the gradients of the cost with respect to both the decision variables
    df_z = 2 * gamma_r_lt * (ZZ - RR) - 2 * gamma_agg * (SS - ZZ) + 2*gamma_rep*(SS - ZZ) + repulsive_gradient ###
    df_s = 2 * gamma_bar * (SS - b) + 2 * gamma_agg * (SS - ZZ) - 2*gamma_rep*(SS - ZZ)

    return f_z, df_z, df_s

#############################################################################
############################ Graph Generation ###############################
#############################################################################

def generate_connected_graph(NN, I_NN, p_ER):

    while True:
        
        #generate adjacency matrix
        Adj = np.random.binomial(1, p_ER, (NN,NN))
        Adj = np.logical_or(Adj,Adj.T)
        Adj = np.multiply(Adj,np.logical_not(I_NN)).astype(int)

        #test for graph's connectivity
        test = np.linalg.matrix_power((I_NN+Adj),NN)
        
        if np.all(test>0):
            print("\nConnected graph has been successfully connected\n")
            break 
        else:
            print("\nThe graph is NOT connected\n")
            quit()
    
    return Adj

def metropolis_hastings(NN, Adj, I_NN):

    '''Assign the Metropolis-Hastings weights for
    each edge of the graph'''

    WW = np.zeros((NN,NN))

    for ii in range(NN):

        N_ii = np.nonzero(Adj[ii])[0] # In-Neighbors of node i
        deg_ii = len(N_ii)
    
        for jj in N_ii:

            N_jj = np.nonzero(Adj[jj])[0] # In-Neighbors of node j
            # deg_jj = len(N_jj)
            deg_jj = N_jj.shape[0]

            #Metropolis-Hastings Weights
            WW[ii,jj] = 1/(1+max( [deg_ii,deg_jj] ))
            # WW[ii,jj] = 1/(1+np.max(np.stack((deg_ii,deg_jj)) ))

    WW += I_NN - np.diag(np.sum(WW,axis=0))

    return WW

#############################################################################
########################## Aggregative Gradient #############################
#############################################################################

def phi_agent(z_i):
  
    '''The phi_i function in the program serves a dual purpose. Firstly, it
    performs an identity mapping on an input vector x_i, returning the vector
    unchanged. Secondly, it facilitates the estimation of the aggregative variable
    (sigma) by providing an identity matrix of the same shape as the input vector.
    This identity matrix is utilized in the calculation of the aggregative variable
    estimate, allowing each agent's decision variable to contribute to the estimation
    process in the aggregative optimization algorithm.'''

    return z_i, np.eye(z_i.shape[0])

def aggregative_gradient(NN, MAXITERS, step_size, n_z, Adj, ZZ, SS, FF, VV, WW, RR, grad_z, grad_s, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep=None, b=None, corr_points=None, avoid_walls=None, targets_motion_type=None, shift_x=None, shift_y=None, local_targets=None, targets_rate_change=None, moving_targets=None, true_barycenter=None, qstar=None, K=None):

    '''This function implements the aggregative optimization algorithm for a multi-robot system.
    It initializes decision variables, estimates of the aggregative variable, and cumulative
    gradients for each agent. Then, it iterates through optimization steps for a specified number
    of iterations. Within each iteration, it updates decision variables based on gradients and
    neighboring agents' information, calculates the estimate of the aggregative variable, and
    accumulates the cost function values. This process guides the collective movement of robots
    towards desired goals, such as the barycenter, while optimizing a defined cost function.'''
    
    def update_local_targets(RR, shift_x, shift_y, targets_motion_type):

        """
        Update the local targets by shifting each point in RR by shift_x in the x-direction
        and shift_y in the y-direction.
        
        Parameters:
            RR (numpy.ndarray): Array of shape (N, 2) containing the coordinates of local targets.
            shift_x (float): Amount to shift in the x-direction.
            shift_y (float): Amount to shift in the y-direction.
            
        Returns:
            numpy.ndarray: Updated array of shape (N, 2) containing the shifted coordinates.
        """
        # Copy the original RR to avoid modifying it in place
        updated_RR = RR.copy()
        
        # Shift each point in RR by the specified amounts
        if targets_motion_type == 'hor':
            
            updated_RR[:, 0] += shift_x  # Shift in the x-direction
            updated_RR[:, 1] += 0  # Shift in the y-direction

        elif targets_motion_type == 'vert':

            updated_RR[:, 0] += 0  # Shift in the x-direction
            updated_RR[:, 1] += shift_y  # Shift in the y-direction

        elif targets_motion_type == 'diag':

            updated_RR[:, 0] += shift_x  # Shift in the x-direction
            updated_RR[:, 1] += shift_y  # Shift in the y-direction

        else:

            print("\nType of local targets'motion not allowed.\n")
            sys.exit()
            
        return updated_RR
    
    print("\n\n---------------------------------------------------------------")
    print("Initial Conditions")
    print("barycenter_repulsion_gain: ", gamma_rep)
    print("local_targets_attraction_gain: ", gamma_r_lt)
    print("barycenter_speed_gain: ", gamma_bar)
    print("barycenter_attraction_gain: ", gamma_agg)
    print("---------------------------------------------------------------")

    #initialize the estimate of the aggregative variable for each agent
    SS[:,0], _ = phi_agent(ZZ[:,0,:])

    #initialize the cost and the cumulative gradient for each agent
    for ii in range(NN):
        
        #initialize the cost function
        f_i, df_z, df_s = cost_function(ZZ[ii,0], SS[ii,0], RR[ii], b, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep, avoid_walls=avoid_walls, corr_points=corr_points, qstar=qstar, K=K)
        FF[0] += f_i
        
        #initialize the cumulative gradient for each agent
        VV[ii,0] = df_s
        grad_z[0] += df_z
        grad_s[0] += df_s

        #initialize true (centralized) quantities
        true_barycenter[0] += phi_agent(ZZ[ii, 0])[0] / NN

    #gradient method for aggregative optimization
    for kk in range (MAXITERS-1): #loop until the maximum number of iterations (put descent here)

        if moving_targets == True:

            #update the local targets
            if (kk % targets_rate_change == 0):
                
                print("Changing local targets!")

                #update local targets
                RR = update_local_targets(RR, shift_x, shift_y, targets_motion_type)

                #append the updated local targets to the list
                local_targets.append(RR.copy())

        #loop over each agent
        for ii in range (NN):

            #obtain the in-neighbors of the current agent based on adjacency
            Nii = np.nonzero(Adj[ii])[0]

            #compute the cost function and its gradient at the current decision variable
            _, df_z, df_s = cost_function(ZZ[ii,kk], SS[ii,kk], RR[ii], b, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep, avoid_walls=avoid_walls, corr_points=corr_points, qstar=qstar, K=K) 
            phi_k, dphi_k = phi_agent(ZZ[ii,kk])

            #update of the agent's decision variable using a steepest descent step
            descent = df_z + dphi_k @ VV[ii,kk]

            #update of the agent's decision variable using a steepest descent step 
            ZZ[ii,kk+1] = ZZ[ii,kk] - step_size * descent
            phi_kp, _ = phi_agent(ZZ[ii,kk+1])

            #update of the estimate (tracker) of the aggregative variable sigma
            SS[ii,kk+1] = WW[ii,ii] * SS[ii,kk] + phi_kp - phi_k

            #compute cost and gradients for update
            for jj in Nii:
                SS[ii, kk+1] += WW[ii,jj]*SS[jj,kk]
            
            f_i, df_z_plus, df_s_plus = cost_function(ZZ[ii,kk+1], SS[ii,kk+1], RR[ii], b, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep, avoid_walls=avoid_walls, corr_points=corr_points, qstar=qstar, K=K) 

            #update of the tracker of the gradient sum
            VV[ii, kk+1] = WW[ii,ii] * VV[ii,kk] + df_s_plus - df_s

            #sum the gradient's neighbors contribution
            for jj in Nii:
                VV[ii,kk+1] += WW[ii,jj] * VV[jj,kk]

            #store the cost function's value
            FF[kk+1] += f_i

            #store the gradients
            grad_z[kk+1] += df_z_plus
            grad_s[kk+1] += df_s_plus

            #store the true barycenter
            true_barycenter[kk+1] += phi_agent(ZZ[ii, kk+1])[0] / NN

        #logs
        if kk % 100 == 0:

            print("\n\n---------------------------------------------------------------")
            print("Aggregative Gradient Algorithm")
            print(f"Completion: {(kk/MAXITERS)*100}%")
            print(f"Total Cost: {FF[kk+1]}")
            print("---------------------------------------------------------------")
    
    print("\n\n---------------------------------------------------------------")
    print("End of Aggregative Tracking.")
    print("---------------------------------------------------------------")
        
    return ZZ, SS, FF, VV, WW, RR, grad_s, grad_z, true_barycenter

def projected_aggregative_tracking(NN, MAXITERS, stepsize, alpha, delta, n_z, Adj, ZZ, SS, FF, VV, WW, RR, grad_z, grad_s, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep=None, walls_avoidance=None, b=None, corr_points=None, walls_margin=None):

    def constraint_set_projection(ZZ, corr_points, walls_margin, kk): #for collision avoidance

        """
        Project the states onto the feasible set, ensuring that the robots stay inside the corridor.

        Parameters:
            ZZ (numpy.ndarray): Array containing the states of the robots.
                                Shape: (NN, iters_per_problem, n_z)
            corr_points (numpy.ndarray): Array containing the coordinates of points defining the corridor.
                                        Shape: (num_points * 2, 2)
            kk (int): Current iteration index.

        Returns:
            numpy.ndarray: Projected states ensuring that the robots remain inside the corridor.
                            Shape: (NN, iters_per_problem, n_z)
        """
        
        #extract x and y coordinates from corr_points
        x_points = [point[0] for point in corr_points]
        y_points = [point[1] for point in corr_points]

        #find the minimum and maximum (x,y) coordinates,add a penalty (margin) for y
        min_x = min(x_points)
        max_x = max(x_points)
        min_y = min(y_points) + walls_margin
        max_y = max(y_points) - walls_margin

        #project the y-coordinate onto the feasible set while walking in the corridor
        if ZZ[0] < max_x: ZZ[1] = np.maximum(np.minimum(ZZ[1], max_y), min_y)

        return ZZ
        
    '''This function implements the aggregative optimization algorithm for a multi-robot system.
    It initializes decision variables, estimates of the aggregative variable, and cumulative
    gradients for each agent. Then, it iterates through optimization steps for a specified number
    of iterations. Within each iteration, it updates decision variables based on gradients and
    neighboring agents' information, calculates the estimate of the aggregative variable, and
    accumulates the cost function values. This process guides the collective movement of robots
    towards desired goals, such as the barycenter, while optimizing a defined cost function.'''

    print("\n\n---------------------------------------------------------------")
    print("Initial Conditions")
    print("barycenter_repulsion_gain: ", gamma_rep)
    print("walls_obstacle_avoidance: ", walls_avoidance)
    print("local_targets_attraction_gain: ", gamma_r_lt)
    print("barycenter_speed_gain: ", gamma_bar)
    print("barycenter_attraction_gain: ", gamma_agg)
    print("---------------------------------------------------------------")

    #initialize each agent's decision variables
    ZZ_init = ZZ[:,0,:]

    #initialize the estimate of the aggregative variable for each agent
    SS[:,0], _ = phi_agent(ZZ_init)

    #initialize the tilde variable
    ZZ_tilde = np.zeros((NN,MAXITERS,n_z))

    #initialize the cost and the cumulative gradient for each agent
    for ii in range(NN):
        
        f_i, df_z, df_s = cost_function(ZZ[ii,0], SS[ii,0], RR[ii], b, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep=gamma_rep)
        FF[0] += f_i
        VV[ii,0] = df_s
        grad_z[0] += df_z
        grad_s[0] += df_s

    for kk in range(MAXITERS-1): #runs over iterations
        
        for ii in range(NN): #loop for each agent

            #obtain the in-neighbors of the current agent based on adjacency
            Nii = np.nonzero(Adj[ii])[0]

            #compute local cost function gradients
            _, df_z, df_s = cost_function(ZZ[ii,kk], SS[ii,kk], RR[ii], b, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep=gamma_rep) 
            phi_k, dphi_k = phi_agent(ZZ[ii,kk])

            if walls_avoidance:

                #compute the state projection on the constraint set
                ZZ_tilde[ii,kk] = constraint_set_projection(ZZ[ii,kk] - alpha * (df_z + dphi_k @ VV[ii,kk]), corr_points, walls_margin, kk)

            else: ZZ_tilde[ii,kk] = ZZ[ii,kk] - alpha * (df_z + dphi_k @ VV[ii,kk])

            #compute the descent direction
            descent = ZZ_tilde[ii,kk] - ZZ[ii,kk]

            #update of the agent's decision variable using a steepest descent step
            ZZ[ii,kk+1] = ZZ[ii,kk] + delta * descent

            #compute phi at the newer decision variable
            phi_kp, _ = phi_agent(ZZ[ii,kk+1])

            #update of the estimate (tracker) of the aggregative variable sigma
            SS[ii,kk+1] = WW[ii,ii] * SS[ii,kk] + phi_kp - phi_k
            for jj in Nii:
                SS[ii, kk+1] += WW[ii,jj]*SS[jj,kk]

            f_i, df_z_plus, df_s_plus = cost_function(ZZ[ii,kk+1], SS[ii,kk+1], RR[ii], b, gamma_r_lt, gamma_bar, gamma_agg, gamma_rep=gamma_rep) 

            #update of the tracker of the gradient sum
            VV[ii, kk+1] = WW[ii,ii]*VV[ii,kk] + df_s_plus - df_s
            for jj in Nii:
                VV[ii,kk+1] += WW[ii,jj]*VV[jj,kk]

            #store the cost function's value
            FF[kk+1] += f_i

            #store the gradients
            grad_z[kk+1] += df_z_plus
            grad_s[kk+1] += df_s_plus
            
        #logs
        if kk % 100 == 0:

            print("\n\n---------------------------------------------------------------")
            print("Online Aggregative Gradient Algorithm")
            print(f"Completion: {(kk/MAXITERS)*100}%")
            print(f"Total Cost: {FF[kk+1]}")
            print("---------------------------------------------------------------")
    
    print("\n\n---------------------------------------------------------------")
    print("End of Online Aggregative Tracking.")
    print("---------------------------------------------------------------")
            
    return ZZ, SS, FF, VV, WW, RR, grad_s, grad_z

#############################################################################
################################ Animation ##################################
#############################################################################

def robots_animation(NN, MAXITERS, ZZ, SS, RR, b, n_z, dt, corr_length=None, corr_width=None, corr_pos=None, local_targets=None, view=None, show_intermediate_targets=None, corr_points=None, show_corr_points=None):

    '''This function generates an animation illustrating the trajectory of multiple robots
    within a multi-robot system over the course of optimization iterations. Given parameters
    such as the number of robots (NN), maximum number of iterations (MAXITERS), and arrays
    representing the positions of robots (XX), estimates of the centroid (SS), and local target
    positions (RR), the function iterates through the optimization process and visualizes the
    movement of robots over time. Each iteration of the animation plots the trajectory, initial
    position, current position, and centroid estimate of each robot, as well as the target
    position of the centroid. This visualization aids in understanding the convergence behavior
    and spatial dynamics of the multi-robot system during optimization.'''

    plt.figure()

    #extract the final targets
    if local_targets: final_targets = local_targets[-1]
    else: final_targets = RR

    if view == 'Static': #initialize the plot limits

        min_x = min(np.min(ZZ[::n_z]), np.min(final_targets[:, 0]))
        max_x = max(np.max(ZZ[::n_z]), np.max(final_targets[:, 0]))
        min_y = min(np.min(ZZ[1::n_z]), np.min(final_targets[:, 1]))
        max_y = max(np.max(ZZ[1::n_z]), np.max(final_targets[:, 1]))
  
    for kk in range(0,MAXITERS,dt):

        # Update plot limits
        if view == 'Static':
            min_x = min(min_x, np.min(ZZ[::n_z, kk]))
            max_x = max(max_x, np.max(ZZ[::n_z, kk]))
            min_y = min(min_y, np.min(ZZ[1::n_z, kk]))
            max_y = max(max_y, np.max(ZZ[1::n_z, kk])) 

        #logs
        if kk % (10 * dt) == 0:

            print("\n\n---------------------------------------------------------------")
            print("Multi-Robot Animation")
            print("View: ", view)
            print(f"Completion: {(kk/MAXITERS)*100}%")
            print("---------------------------------------------------------------")

        #plot barycenter's target position
        plt.plot(b[0],b[1], marker='*', markersize=10, color = 'tab:orange')

        #plot the corridor
        if corr_pos is not None:
            
            #plot the corridor in space
            plot_corridor(corr_length, corr_width, corr_pos)

            #plot the discrete corridor points
            if show_corr_points == True:

                #Circles
                x_points = [point[0] for point in corr_points]
                y_points = [point[1] for point in corr_points]
                plt.plot(x_points, y_points, marker='o', linestyle='', color='blue')

                for i in range(len(corr_points)):
                    circle = plt.Circle(corr_points[i], 3, color='blue', fill=False, label='Circle')
                    plt.gca().add_patch(circle)
        
        #plot both intermediate and final targets
        if show_intermediate_targets:

            #iterate through local targets and plot them with different marker sizes
            for i, tar in enumerate(local_targets):
                
                #plot first and last markers with higher sizes
                if i == len(local_targets) - 1: plt.scatter(tar[:, 0], tar[:, 1], marker='x', s=110, color="red")
                    
                #plot intermediate markers with smaller sizes
                else: plt.scatter(tar[:, 0], tar[:, 1], marker='x', s=15, color="salmon")

        #plot the final target positions
        else: plt.scatter(final_targets[:,0], final_targets[:,1], marker = 'x', color = 'red')
        
        for ii in range(NN):

            #plot trajectory of agent i 
            plt.plot(ZZ[ii,:,0].T, ZZ[ii,:,1].T, linestyle='--', color = 'tab:blue',alpha=0.3)
            
            #plot initial position of agent i
            plt.plot(ZZ[ii,0,0], ZZ[ii,0,1], marker='o', markersize=10, color = 'tab:blue',alpha=0.3)

            #plot current position of agent i at time t 
            plt.plot(ZZ[ii,kk,0],ZZ[ii,kk,1], marker='o', markersize=10, color = '#1f77b4')

            #plot estimate of agent i of the centroid (barycenter) at time t
            plt.plot(SS[ii,kk,0],SS[ii,kk,1], marker='.', markersize=5, color = 'tab:red')

        if view == 'Static': #static viewpoint
            plt.xlim(min_x - 1, max_x + 1)
            plt.ylim(min_y - 1, max_y + 1)

        else:
            axes_lim = (np.min(ZZ)-1,np.max(ZZ)+1)
            plt.xlim(axes_lim); plt.ylim(axes_lim)

        plt.axis('equal')     
        plt.show(block=False)
        plt.pause(0.05)
        if kk < MAXITERS - dt - 1: plt.clf()

    print("\n\n---------------------------------------------------------------")
    print("End of Multi-Robot Animation.")
    print("---------------------------------------------------------------")

def generate_circular_positions(NN, n_z, central_target_pos, radius, gain):

    """
    Create target positions for leaders around a central point.
    
    Parameters:
        N (int): Number of agents
        central_position (tuple): Central position around which target positions are created (x, y)
        radius (float): Radius of the circle around the central position
        gain (float): Gain controlling the degree of closeness of the leaders to the central point
        
    Returns:
        target_positions (numpy.ndarray): Array of target positions for agents
    """
      
    target_positions = np.zeros((NN, n_z))

    #generate the target positions around the central point
    for i in range(NN):

        #define a target angle for the i-th robot
        angle = (2 * np.pi * i) / NN
        
        #find the offset coordinates via trigonometry
        x_offset = radius * np.cos(angle)
        y_offset = radius * np.sin(angle)

        #store the location of the i-th robot
        if n_z == 2: target_positions[i] = (central_target_pos[0] + gain * x_offset, central_target_pos[1] + gain * y_offset)
        else: target_positions[i, :2] = (central_target_pos[0] + gain * x_offset, central_target_pos[1] + gain * y_offset)

    return target_positions

def random_initialization(NN, n_z, corr_points):

    """
    Initializes NN initial positions within a specified area.

    Parameters:
    NN (int): Number of initial positions.
    upper_left (tuple): Coordinates of the upper left point defining the area (x, y).
    lower_left (tuple): Coordinates of the lower left point defining the area (x, y).

    Returns:
    initial_positions (numpy.ndarray): Array of shape (NN, 2) containing initial positions.
    """

    #initialize
    target_positions = np.zeros((NN, n_z))
    
    #extract the entrance points
    upper_left = corr_points[0]  #first point in the list
    lower_left = corr_points[1]  #second point in the list

    #assign target positions randomly
    target_positions[:, 0] = np.random.uniform(0, upper_left[0]-1, NN)  #x coordinates
    target_positions[:, 1] = np.random.uniform(lower_left[1]-2, upper_left[1]+2, NN)  #y coordinates

    return target_positions