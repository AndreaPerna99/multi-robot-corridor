#DAS Project 2023/24
#Task 2: centralized_animation.py
#Gianluca Di Mauro, Andrea Perna, Meisam Tavakoli

######################################################################
######################### Libraries ##################################
######################################################################

import numpy as np
import matplotlib.pyplot as plt
import signal
import yaml
import os
import pandas as pd
import plotly.graph_objects as go
signal.signal(signal.SIGINT, signal.SIG_DFL)

######################################################################
######################### Functions ##################################
######################################################################

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
    #if local_targets: final_targets = local_targets[-1]
    #else: 
    final_targets = RR

    local_targets = RR
    #final_targets = RR
    
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

            #iterate through local targets and plot them with different marker sizes - for different colours, don't set color parameter
            for i, tar in enumerate(local_targets):
                
                #plot first and last markers with higher sizes
                if i == len(local_targets) - 1:  #or i = 0, last markers
                    plt.scatter(tar[:, 0], tar[:, 1], marker='x', s=110, color="red")
                    
                #plot intermediate markers with smaller sizes
                else:
                    plt.scatter(tar[:, 0], tar[:, 1], marker='x', s=15, color="salmon")

        #plot the final target positions
        else:
            #plt.scatter(final_targets[:,0], final_targets[:,1], marker = 'x', color = 'red')
            #iterate through local targets and plot them with different marker sizes - for different colours, don't set color parameter
            for i, tar in enumerate(local_targets):
                plt.scatter(tar[:, 0], tar[:, 1], marker='x', s=110, color="red")
                
        for ii in range(NN):

            #plot trajectory of agent i 
            plt.plot(ZZ[ii,:,0].T,ZZ[ii,:,1].T, linestyle='--', color = 'tab:blue',alpha=0.3)
            
            #plot initial position of agent i
            plt.plot(ZZ[ii,0,0],ZZ[ii,0,1], marker='o', markersize=10, color = 'tab:blue',alpha=0.3)

            #plot current position of agent i at time t 
            plt.plot(ZZ[ii,kk,0],ZZ[ii,kk,1], marker='o', markersize=10, color = '#1f77b4')

            #plot estimate of agent i of the centroid (barycenter) at time t
            plt.plot(SS[ii,kk,0],SS[ii,kk,1], marker='.', markersize=5, color = 'tab:red')

        #plot settings
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

def plot_agents_data(NN, MAXITERS, SS, YY, ZZ, RR, FF, grad_z, grad_s):

    #create a vector for the number of iterations
    iter_points = np.arange(MAXITERS)

    ######################################################
    ##################### 3D Plot ########################
    ######################################################

    #create a 3D scatter plot
    go_fig = go.Figure()
    
    for ii in range(NN):
        
        #extract the (x,y) agent's position over time
        ZZ_x = ZZ[ii, :, 0]
        ZZ_y = ZZ[ii, :, 1]

        #add scatter plot for the current agent's position
        go_fig.add_trace(go.Scatter3d(
            x = iter_points, y = ZZ_x, z = ZZ_y,
            mode='lines+markers',
            name=f'Agent {ii}',
            marker=dict(size=4),
            line=dict(width=2)
        ))
    
        #extract the barycenter coordinates over time
        SS_x = SS[ii, :, 0]
        SS_y = SS[ii, :, 1]

        #add scatter plot for the barycenter
        go_fig.add_trace(go.Scatter3d(
            x = iter_points, y = SS_x, z = SS_y,
            mode='lines+markers',
            name=f'Barycenter {ii}',
            marker=dict(size=1),
            line=dict(width=1)
        ))
        
        #extract the final target
        RR_x = RR[ii, MAXITERS-1, 0]
        RR_y = RR[ii, MAXITERS-1, 1]

        #add the agent's target to the plot
        go_fig.add_trace(go.Scatter3d(
            x=[MAXITERS-1],
            y=[RR_x],
            z=[RR_y],
            mode='markers',
            name=f'Target_{ii}:(x={RR_x}, y={RR_y})',
            marker=dict(size=8, color='red', symbol='cross')
        ))

    #add scatter plot for the current agent
    go_fig.add_trace(go.Scatter3d(
        x = iter_points, y = ZZ_x, z = ZZ_y,
        mode='lines+markers',
        name=f'Agent {ii}',
        marker=dict(size=4),
        line=dict(width=2)
    ))

    #set labels and title
    go_fig.update_layout(

        title='Agents Dynamics Over Time',
        scene=dict(
            xaxis_title='Time',
            yaxis_title='X Position',
            zaxis_title='Y Position'
        ),
        legend=dict(title='Agents')
    )

    #show plot
    go_fig.show()

    ######################################################
    ################# Cost Function Plot #################
    ######################################################

    #initialize the cost
    cost = np.zeros(MAXITERS)

    #compute the centralized cumulative cost
    for kk in range(MAXITERS):

        #loop over the agents
        for ii in range(NN):    

            #compute the cumulative cost at kk        
            cost[kk] += FF[ii, kk]

    #plot the cost function
    plt.figure()
    #plt.plot(np.arange(MAXITERS), np.abs(cost-cost[-1]), '--', linewidth=3)
    plt.plot(np.arange(MAXITERS), cost, '--', linewidth=3)
    plt.yscale('log')        
    plt.xlabel(r"Iterations $t$")
    plt.ylabel(r"Cost")
    plt.title("Evolution of the cost function")
    plt.grid()
    plt.show()

    ######################################################
    ################# Gradient Norm Plot #################
    ######################################################

    #initialize the gradient norm
    grad_norm_z = np.zeros(MAXITERS)
    grad_norm_s = np.zeros(MAXITERS)

    #compute the centralized gradient norm
    for kk in range(MAXITERS):

        #loop over the agents
        for ii in range(NN):    

            #compute the gradient norm of the decision variable at kk    
            grad_norm_z[kk] += np.linalg.norm(grad_z[ii, kk, :])

            #compute the gradient norm of the barycenter at kk    
            grad_norm_s[kk] += np.linalg.norm(grad_s[ii, kk, :])

    #plot the gradient norm of agent's positions
    plt.figure()
    #plt.plot(np.arange(MAXITERS), np.abs(cost-cost[-1]), '--', linewidth=3)
    plt.plot(np.arange(MAXITERS), grad_norm_z, '--', linewidth=3)
    plt.yscale('log')        
    plt.xlabel(r"Iterations $t$")
    plt.ylabel(r"ZZ Gradient Norm")
    plt.title("Evolution of the agents' position gradient norm")
    plt.grid()
    plt.show()

    #plot the gradient norm of team's barycenter
    plt.figure()
    #plt.plot(np.arange(MAXITERS), np.abs(cost-cost[-1]), '--', linewidth=3)
    plt.plot(np.arange(MAXITERS), grad_norm_s, '--', linewidth=3)
    plt.yscale('log')        
    plt.xlabel(r"Iterations $t$")
    plt.ylabel(r"SS Gradient Norm")
    plt.title("Evolution of the team's barycenter gradient norm")
    plt.grid()
    plt.show()

def generate_corridor_points(corr_length, corr_width, corr_pos):

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

    for point in range(num_points2):

        x_p = points[0][0]
        y_ph = points[0][1] + point
        points.append((x_p, y_ph))

    for point in range(num_points2):
        x_pl = points[1][0]
        y_pl = points[1][1] - point
        points.append((x_pl, y_pl))   

    return points

#############################################################################
############################## YAML Parameters ##############################
#############################################################################

#define the task
task = 2.3

#define the yaml file for parameters' extraction
file_name_yaml = 'parameters_task_2_3.yaml'

#get the current directory of the script
current_dir = os.path.dirname(os.path.realpath(__file__))

#construct the relative path to the folder to access
yaml_folder_path = os.path.join(current_dir, '..', 'parameters')

#define the yaml file's path
yaml_file_path = os.path.join(yaml_folder_path, file_name_yaml)

#get data from YAML file
if os.path.exists(yaml_file_path):

    #read map's information
    with open(yaml_file_path, 'r') as file: yaml_data = yaml.safe_load(file)

    #extract agents' parameters
    NN = yaml_data['NN']
    n_z = yaml_data['n_z']

    #extract algorithm's parameters
    MAXITERS = yaml_data['MAXITERS']
    MAXITERS += 1
    step_size = yaml_data['step_size']
    dt = yaml_data['dt']

    #extract visualization's parameters
    visu_frequency = yaml_data['visu_frequency']
    
    #extract targets' parameters
    b = yaml_data['b']
    target_avg = yaml_data['target_avg']
    initial_avg = yaml_data['initial_avg']
    radius = yaml_data['radius']
    gain = yaml_data['gain']

    #extract corridor's parameters
    corr_length = yaml_data['corr_length']
    corr_width = yaml_data['corr_width']
    corr_pos = yaml_data['corr_pos']
    corr_bound_offset = yaml_data['corr_bound_offset']
    walls_margin = yaml_data['walls_margin']
    num_corr_points = yaml_data['num_corr_points']
    show_corr_points = yaml_data['show_corr_points']

else: print(f"[DATA_EXTRACTION]: Yaml file '{file_name_yaml}' not found in folder '{yaml_folder_path}'")

#############################################################################
############################## Corridor Points ##############################
#############################################################################

#generate points
corr_points = generate_corridor_points(corr_length, corr_width, corr_pos)

#transform corr_points into an array
corr_points = np.array(corr_points)

#############################################################################
############################## CSV Parameters ###############################
#############################################################################

#initialize the list of dataframes
dataframes = []

#define the dataframe's content
names=['ZZ_x', 'ZZ_y', 'SS_x', 'SS_y', 'YY_x', 'YY_y', 'RR_x', 'RR_y', 'FF', 'grad_z_x', 'grad_z_y', 'grad_s_x', 'grad_s_y']

#initialize containers for the agents' data
ZZ = np.zeros((NN, MAXITERS, n_z))
SS = np.zeros((NN, MAXITERS, n_z))
YY = np.zeros((NN, MAXITERS, n_z))
RR = np.zeros((NN, MAXITERS, n_z))
FF = np.zeros((NN, MAXITERS))
grad_z = np.zeros((NN, MAXITERS, n_z))
grad_s = np.zeros((NN, MAXITERS, n_z))

#get the path to csv's folder
csv_folder_path = os.path.join(current_dir, '_csv_file')

#open all the agents'files
for ii in range(NN):
    
    #create a list of file names
    csv_file_path = os.path.join(csv_folder_path, "agent_{}.csv".format(ii))
    
    #create a dataframe for the agent's data
    df_i = pd.read_csv(csv_file_path, header=None, names=names)

    #store the agent's dataframe
    dataframes.append(df_i)

#extract data from csv file and gather them
for ii, df_i in enumerate(dataframes):

    SS[ii] = df_i[['SS_x', 'SS_y']]
    YY[ii] = df_i[['YY_x', 'YY_y']]
    ZZ[ii] = df_i[['ZZ_x', 'ZZ_y']]
    RR[ii] = df_i[['RR_x', 'RR_y']]
    FF[ii] = df_i[['FF']].squeeze()
    grad_z[ii] = df_i[['grad_z_x', 'grad_z_y']]
    grad_s[ii] = df_i[['grad_s_x', 'grad_s_y']]

#############################################################################
######################### Centralized Animation #############################
#############################################################################

#plot cost and gradient
plot_agents_data(NN, MAXITERS, SS, YY, ZZ, RR, FF, grad_z, grad_s)

#launch the robots' animation
robots_animation(NN, MAXITERS, ZZ, SS, RR, b, n_z, dt, corr_length, corr_width, corr_pos, local_targets=RR, view=None, corr_points=corr_points, show_corr_points=show_corr_points)
