#DAS Project 2023/24
#Task 2: task2_ros.launch.py
#Gianluca Di Mauro, Andrea Perna, Meisam Tavakoli

######################################################################
######################### Libraries ##################################
######################################################################

from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
import os
import numpy as np
import yaml
import time
import networkx as nx

######################################################################
######################### Parameters #################################
######################################################################

#define the task
task = 2.1

#define the yaml file for parameters' extraction
if task == 2.1: file_name_yaml = 'parameters_task_2_1.yaml'
elif task == 2.3: file_name_yaml = 'parameters_task_2_3.yaml'

#define the path to the parameters' file
parameters_folder = 'src/task2_ros/parameters'
yaml_path = os.path.join(parameters_folder, file_name_yaml)

#get data from YAML file
if os.path.exists(yaml_path):

    #read map's information
    with open(yaml_path, 'r') as file: yaml_data = yaml.safe_load(file)

    #extract agents' parameters
    NN = yaml_data['NN']
    n_z = yaml_data['n_z']

    #extract graph parameters
    p_ER = yaml_data['p_ER']

    #extract algorithm's parameters
    MAXITERS = yaml_data['MAXITERS']
    step_size = yaml_data['step_size']
    dt = yaml_data['dt']

    #extract cost function's gains
    gamma_r_lt = yaml_data['gamma_r_lt']
    gamma_agg = yaml_data['gamma_agg']
    gamma_bar = yaml_data['gamma_bar']
    gamma_rep = yaml_data['gamma_rep']

    #extract visualization's parameters
    visu_frequency = yaml_data['visu_frequency']
    timer_period = yaml_data['timer_period']
    
    #extract targets' parameters
    b = yaml_data['b']
    target_avg = yaml_data['target_avg']
    initial_avg = yaml_data['initial_avg']
    radius = yaml_data['radius']
    gain = yaml_data['gain']
    targets_motion_type = yaml_data['targets_motion_type']
    moving_local_targets = yaml_data['moving_local_targets']
    targets_rate_change = yaml_data['targets_rate_change']
    targets_shift_x = yaml_data['targets_shift_x']
    targets_shift_y = yaml_data['targets_shift_y']

    #extract corridor's parameters
    corr_length = yaml_data['corr_length']
    corr_width = yaml_data['corr_width']
    corr_pos = yaml_data['corr_pos']
    corr_bound_offset = yaml_data['corr_bound_offset']
    walls_margin = yaml_data['walls_margin']
    num_corr_points = yaml_data['num_corr_points']
    avoid_walls = yaml_data['avoid_walls']

else: print(f"[DATA_EXTRACTION]: Yaml file '{file_name_yaml}' not found in folder '{parameters_folder}'")

######################################################################
########################## Functions #################################
######################################################################

def generate_connected_graph(NN, I_NN, p_ER):

    while True:
        
        #generate adjacency matrix
        Adj = np.random.binomial(1, p_ER, (NN,NN))
        Adj = np.logical_or(Adj,Adj.T)
        Adj = np.multiply(Adj,np.logical_not(I_NN)).astype(int)

        #test for graph's connectivity
        test = np.linalg.matrix_power((I_NN+Adj),NN)
        
        if np.all(test>0):
            print("\nThe graph has been successfully connected\n")
            break 
        
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

def generate_corridor_points(corr_length, corr_width, corr_pos, num_points):

    """Generate points along the edges of the corridor."""
    x_pos, y_pos = corr_pos
    half_width = corr_width / 2
    points = []
    
    # Generate points along both upper and lower edges
    for i in range(num_points):

        # Upper edge point
        x_upper = x_pos - corr_length / 2 + (i / (num_points - 1)) * corr_length
        y_upper = y_pos + half_width
        points.append((x_upper, y_upper))
        
        # Lower edge point
        x_lower = x_pos - corr_length / 2 + (i / (num_points - 1)) * corr_length
        y_lower = y_pos - half_width
        points.append((x_lower, y_lower))

    return points

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
Adj = generate_connected_graph(NN, I_NN, p_ER)

#assign the Metropolis-Hastings weights to the graph's edges
WW = metropolis_hastings(NN, Adj, I_NN)

#convert to list of strings to be passed as parameters
Adj_str = np.array2string(Adj, separator=",")
WW_str = np.array2string(WW, separator=',')

######################################################################
########################### Initialization ###########################
######################################################################

#initialize the robots randomly in a radial way
ZZ_init = generate_circular_positions(NN, n_z, initial_avg, radius, gain)

#create the local targets for the agents
RR = generate_circular_positions(NN, n_z, target_avg, radius, gain)

#find the corridor's points
corr_points = generate_corridor_points(corr_length, corr_width, corr_pos, num_corr_points)

#transform corr_points to make them feasible for message passing
corr_points = str(corr_points)

#define the corridor's entrance and exit
corr_entrance = [corr_pos[0] - corr_length / 2 - corr_bound_offset, corr_pos[1]]
corr_exit = [corr_pos[0] + corr_length / 2 + corr_bound_offset, corr_pos[1]]

######################################################################
########################### Nodes Launch #############################
######################################################################

#show the title
message_handler("title", task)

def generate_launch_description():

    '''The generate_launch_description function builds a LaunchDescription
    for ROS 2, primarily focusing on setting up visualization and computation 
    nodes for a multi-robot system. It first configures RViz for visualization
    and then iterates over each agent, launching nodes for both computation
    and visualization. Computation nodes receive various parameters like agent
    ID, neighbors, initial positions, and algorithm parameters. Visualization
    nodes simply receive the agent's ID and frequency. Finally, it returns the
    assembled LaunchDescription to initialize and run the simulation environment.'''

    launch_description = [] # Append here your nodes

    ###########################################################
    ######################### RVIZ ############################
    ###########################################################

    # initialize launch description with rviz executable
    rviz_config_dir = get_package_share_directory('task2_ros')
    rviz_config_file = os.path.join(rviz_config_dir, 'rviz_config.rviz')

    #launch the rviz visualization tool
    launch_description.append(
        Node(
            package='rviz2', 
            executable='rviz2', 
            arguments=['-d', rviz_config_file],
            output='screen',
            ))
 
    ###########################################################
    ######################### AGENTS ##########################
    ###########################################################

    for ii in range(NN):

        #find the number of neighbors
        N_i = np.nonzero(Adj[:, ii])[0].tolist()

        #launch the agents'nodes for computation
        launch_description.append(
            Node(
                package='task2_ros',
                namespace =f'agent_{ii}',
                executable='the_agent',
                
                parameters=[{ # dictionary for collecting parameters
                                'agent_id': ii, 
                                'neigh': N_i,
                                'task': task,
                                'initial_state': ZZ_init[ii].tolist(),
                                'max_iters': MAXITERS,
                                'timer_period': timer_period,
                                'loc_targets_gain': gamma_r_lt,
                                'barycenter_agg_gain': gamma_agg,
                                'barycenter_goal_gain': gamma_bar,
                                'barycenter_repulsion_gain': gamma_rep,
                                'step_size': step_size,
                                'barycenter_goal': b,
                                'agent_local_target': RR[ii].tolist(),
                                'n_z': n_z,
                                'adjacency_matrix': Adj_str,
                                'weights_matrix': WW_str,
                                'corr_length': corr_length,
                                'corr_width': corr_width,
                                'corr_position': corr_pos,
                                'corr_points': corr_points,
                                'corr_bound_offset': corr_bound_offset,
                                'walls_margin': walls_margin,
                                'targets_motion_type': targets_motion_type,
                                'moving_local_targets': moving_local_targets,
                                'targets_shift_x': targets_shift_x,
                                'targets_shift_y': targets_shift_y,
                                'targets_rate_change': targets_rate_change,
                                }],
                
                output='screen',
                prefix=f'xterm -title "agent_{ii}" -hold -e',
            ))

        #launch the agents'nodes for plotting
        launch_description.append(
            Node(
                package='task2_ros', 
                namespace='agent_{}'.format(ii),
                executable='visualizer', 
                parameters=[{
                                'agent_id': ii,
                                'task': task,
                                'node_frequency': visu_frequency,
                                }],
                output='screen',
            ))

    return LaunchDescription(launch_description)
