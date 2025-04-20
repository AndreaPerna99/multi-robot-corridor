#DAS Project 2023/24
#Task 2: the_agent.py
#Gianluca Di Mauro, Andrea Perna, Meisam Tavakoli

######################################################################
######################### Libraries ##################################
######################################################################

from time import sleep
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray as MsgFloat
from std_msgs.msg import String
import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
import os
#import Project_Functions as funcs
import time
import sys
import ast
np.random.seed(0)

######################################################################
############################# Class ##################################
######################################################################

class Agent(Node):

    ######################################################################
    ######################### Initialization #############################
    ######################################################################

    def __init__(self):

        #allow parameters passing from the launch file
        super().__init__("agent", allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True,)        

        '''The __init__ method initializes the agent's parameters by retrieving values from a launch file in a ROS system.
        These parameters include the agent's ID, initial state, number of dimensions, neighboring agents, adjacency matrix,
        weights matrix, various gains for optimization, maximum iterations, step size, corridor dimensions, and wall margins.
        It transforms certain parameters back into NumPy arrays and initializes trackers for the agent's cost and decision
        variables. Additionally, it sets up subscriptions to messages from neighboring agents, creates publishers for message
        passing, sets up a timer for callbacks, initializes a dictionary to store received messages from neighbors, and logs
        initial conditions for monitoring.'''

        ######################################################
        ############### Launch File Parameters ###############
        ######################################################

        #get agent's parameters from launch file
        self.agent_id = self.get_parameter('agent_id').value
        self.ZZ_init = np.array(self.get_parameter('initial_state').value)
        self.n_z = self.get_parameter('n_z').value
        self.task = self.get_parameter('task').value

        #get graph's parameters from launch file
        self.neigh = self.get_parameter('neigh').value
        self.Adj_str = self.get_parameter('adjacency_matrix').value
        self.WW_str = self.get_parameter('weights_matrix').value

        #get targets' parameters from launch file
        self.gamma_r_lt = self.get_parameter('loc_targets_gain').value
        self.gamma_agg = self.get_parameter('barycenter_agg_gain').value
        self.gamma_bar = self.get_parameter('barycenter_goal_gain').value
        self.gamma_rep = self.get_parameter('barycenter_repulsion_gain').value
        self.RR = np.array(self.get_parameter('agent_local_target').value)
        self.b = np.array(self.get_parameter('barycenter_goal').value)
        self.targets_motion_type = self.get_parameter('targets_motion_type').value
        self.moving_local_targets = self.get_parameter('moving_local_targets').value
        self.targets_rate_change = self.get_parameter('targets_rate_change').value
        self.shift_x = self.get_parameter('targets_shift_x').value
        self.shift_y = self.get_parameter('targets_shift_y').value

        #get algorithm's parameters from launch file
        self.MAXITERS = self.get_parameter('max_iters').value
        self.step_size = self.get_parameter('step_size').value
        
        #get visualization's parameters from launch file
        self.timer_period = self.get_parameter('timer_period').value

        #get corridor's parameters from the launch file
        self.corr_length = self.get_parameter('corr_length').value
        self.corr_width = self.get_parameter('corr_width').value
        self.corr_pos = self.get_parameter('corr_position').value
        self.bound_offset = self.get_parameter('corr_bound_offset').value
        self.walls_margin = self.get_parameter('walls_margin').value
        self.avoid_walls = self.get_parameter('avoid_walls').value
        self.corr_points = self.get_parameter('corr_points').value #############

        #parse the string to extract the list representation
        WW_list = ast.literal_eval(self.WW_str)
        Adj_list = ast.literal_eval(self.Adj_str)
        WW_shape = (len(WW_list), len(WW_list[0]))
        Adj_shape = (len(Adj_list), len(Adj_list[0]))
        
        #transform back Adj and WW to np.ndarrays
        self.Adj = np.array(eval(self.Adj_str)).reshape(Adj_shape)
        self.WW = np.array(eval(self.WW_str)).reshape(WW_shape)

        ######################################################
        ################### Initialization ###################
        ######################################################

        #define other parameters
        self.degree = len(self.neigh)
        self.kk = 0
        self.local_targets = []

        #initialize containers keeping track of changes over iterations
        self.ZZ_overall = np.zeros((self.MAXITERS, self.n_z))
        self.FF_overall = np.zeros((self.MAXITERS))
        self.SS_overall = np.zeros((self.MAXITERS, self.n_z)) #aggregative variable (sigmax)
        self.VV_overall = np.zeros((self.MAXITERS, self.n_z)) #sum (nabla2)
     
        #initialize the decision variables
        ZZ = self.ZZ_init
        self.ZZ = np.array(ZZ)

        #initialize the estimate of the aggregative variable for each agent
        SS, _ = self.phi_agent(self.ZZ)
        self.SS = np.array(SS)

        #initialize the cumulative gradient for each agent
        f_i_, df_z, df_s = self.cost_function(self.ZZ, self.SS, self.RR, self.b)
        self.VV = np.array(df_s)

        #initialize cost and gradients
        self.FF = f_i_
        self.grad_z = df_z
        self.grad_s = df_s

        ######################################################
        ############# Publishers & Subscribers ###############
        ######################################################

        #initialize dictionary for subscriptions
        self.subscriptions_list = {}

        #create a subscription for each neighbor
        for j in self.neigh: self.subscriptions_list[j] = self.create_subscription(MsgFloat, '/topic_{}'.format(j), lambda msg, node = j: self.listener_callback(msg, node), 10)
        
        #create the publishers for message passing and visualization, respectively
        self.publisher = self.create_publisher(MsgFloat, f"/topic_{self.agent_id}", 10)
            
        #creates the timer for callback
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        #initialize a dictionary with a list of received messages from each neighbor j [a queue]
        self.received_data = { j:[] for j in self.neigh }

        ######################################################
        ############# Centralized Visualization ##############
        ######################################################

        #get the current directory of the script
        current_dir = os.path.dirname(os.path.realpath(__file__))

        #construct the relative path to the folder to access
        csv_path = os.path.join(current_dir, '..', 'centralized_animation', '_csv_file')

        #create logging file
        self.file_name = os.path.join(csv_path, "agent_{}.csv".format(self.agent_id))
        
        try:
            # Create the file
            with open(self.file_name, "w+") as file:
                file.write("This is a test message.")

        except OSError as e:
            print(f"Error occurred while creating file: {e}")

        file = open(self.file_name, "w+") # 'w+' needs to create file and open in writing mode if doesn't exist
        file.close()

        ######################################################
        ######################### Logs #######################
        ######################################################

        #logs
        print("\n\n---------------------------------------------------------------")
        print(f"Setup of agent {self.agent_id} completed")
        print("agent_id: ", self.agent_id)
        print("n_x: ", self.n_z)
        print("XX_init: ", self.ZZ_init)
        print("Adj: ", self.Adj)
        print("WW: ", self.WW_str)
        print("number of neighbors: ", self.neigh)
        print("barycenter_repulsion_gain: ", self.gamma_rep)
        print("local_targets_attraction_gain: ", self.gamma_r_lt)
        print("barycenter_speed_gain: ", self.gamma_bar)
        print("barycenter_attraction_gain: ", self.gamma_agg)
        print("barycenter_target: ", self.b)
        print("local target goal: ", self.RR)
        print("---------------------------------------------------------------")

    ######################################################################
    ######################### Message Passing ############################
    ######################################################################

    def listener_callback(self, msg, node):

        '''
        This callback should store the message read in the topic from the
        specific neighbors. The message has the attribute data, which is a list.
        The first element of the agent's message is defined to be an integer,
        which tells who the agent is. Then, the buffer is used to store the content.
        '''

        self.received_data[node].append(list(msg.data))
        return None
    
    def csv_store(self, file_name, string):

        """
        This function allows writing agent's data information inside
        the csv file, in order to make centralized visualization possible.
        """

        file = open(file_name, "a") # "a" is for append
        file.write(string)
        file.close()

    def publish_message(self):

        '''The publish_message method formats and sends essential agent data, such as its ID, iteration count, aggregative variable,
        cumulative gradient, and decision variable, as a ROS-compatible message to a designated topic. Additionally, it logs this
        information for monitoring purposes.'''
        
        ######################################################
        ################# Message Publication ################
        ######################################################
        
        #create the communication message
        msg = MsgFloat()

        #fill the message with agent's information
        msg.data = [float(self.kk)] + self.ZZ.tolist() + self.SS.tolist() + self.VV.tolist() + self.RR.tolist() + [float(self.FF)] + self.grad_z.tolist() + self.grad_s.tolist()
            
        #publish the message
        self.publisher.publish(msg)

        ######################################################
        ############## Centralized Visualization #############
        ######################################################
                
        #save data on csv file
        data_for_csv = msg.data.tolist().copy()
        data_for_csv = [str(round(element,4)) for element in data_for_csv[1:]] #round each element of the list to four decimal places, skip first element
        data_for_csv = ','.join(data_for_csv)
        self.csv_store(self.file_name,data_for_csv+'\n')

        ######################################################
        ###################### Logs ##########################
        ######################################################

        #show the communication messege's content
        SS_i_string = f"{np.array2string(self.SS, precision=4, floatmode='fixed', separator=', ')}"
        VV_i_string = f"{np.array2string(self.VV, precision=4, floatmode='fixed', separator=', ')}"
        ZZ_i_string = f"{np.array2string(self.ZZ, precision=4, floatmode='fixed', separator=', ')}"

        print("\n\n---------------------------------------------------------------")
        print("Aggregative Gradient Algorithm")
        self.get_logger().info( f"Iter: {self.kk}, agent_{self.agent_id}: SS_i:{SS_i_string}, VV_i:{VV_i_string}, FF_i:{self.FF}: ZZ_i:{ZZ_i_string}, RR_i:{self.RR}")
        print("---------------------------------------------------------------")

    ######################################################################
    ######################### Cost Function ##############################
    ######################################################################

    def cost_function(self, ZZ, SS, RR, b):
    
        """
        Defines the cost function that needs to be minimized from the distributed aggregative optimization algorithm.

        Inputs:
            z_i: Decision variable of the current agent.
            sigma: Aggregative variable estimate.
            r_i: Reference position of the current agent.
            b: Target position of the barycenter.
            gamma_r_lt: Weight for the distance between the robot and its local target.
            gamma_bar: Weight for the distance between the estimated barycenter and its target.
            gamma_agg: Weight for the distance between the robot and the team's estimated barycenter.
            gamma_rep: Weight for the distance between the robot and the opposite of the barycenter.
            ZZ: Matrix of decision variables for all agents.
            Adj: Adjacency matrix indicating neighboring agents.

        Returns:
            f_i: Value of the cost function evaluated for current x_i and sigma.
            df_z: Gradient of the cost function wrt decision variable x_i of current agent.
            df_s: Gradient of the cost function wrt the aggregative decision variable sigma.
        """

        def potential_field():

            '''Remember to add also the vertical walls points and to remove multi-optimization
            The potential_field function computes the repulsive potential field for a robot
            navigating a corridor. It iterates through corridor points, calculates the distance
            from the robot, and computes the potential and gradient based on a distance threshold.
            Finally, it returns the total potential and gradient.'''

            #potential field constants
            q_star = 4
            K = 200

            #compute repulsive potential field for corridor points
            repulsive_potential = 0
            repulsive_gradient = np.zeros_like(ZZ)

            #compute repulsive potential field for each corridor point
            for point in self.corr_points:

                #compute the squared distance between the robot and the corridor's point
                d_point = (ZZ - np.array(point)) @ (ZZ - np.array(point))

                if d_point <= q_star:

                    repulsive_potential += 0.5 * K * (1 / d_point - 1 / q_star) ** 2
                    repulsive_gradient += K * (1/ q_star - 1/d_point) * (2*(ZZ - np.array(point)) / (d_point**2))
                
                else:

                    repulsive_potential += 0
                    repulsive_gradient += 0
        
            return repulsive_potential, repulsive_gradient
    
        #set zero for None gains
        if self.gamma_rep is None: self.gamma_rep = 0

        #compute repulsive potential dield term for each corridor's point
        if self.avoid_walls: repulsive_potential, repulsive_gradient = potential_field()
        else:
            repulsive_potential = 0
            repulsive_gradient = 0
            
        # Define cost sub-functions
        d_robot_target = np.linalg.norm(ZZ - RR) ** 2  # Distance between robot (x_i) and its local target (r_i)
        d_barycenter_target = np.linalg.norm(SS - b) ** 2  # Distance between estimated barycenter (sigma) and its target (b)
        d_robot_barycenter = np.linalg.norm(SS - ZZ) ** 2  # Distance between robot (x_i) and team's estimated barycenter (sigma)

        # Compute distance of each robot from the center
        radial_distance = np.linalg.norm(SS - ZZ)
        radial_penalty = radial_distance ** 2  # Adjust the power to control the strength of the penalty

        #define the cost function for agent i to be minimized
        f_z = self.gamma_r_lt * d_robot_target + self.gamma_bar * d_barycenter_target + self.gamma_agg * d_robot_barycenter + self.gamma_rep * radial_penalty + repulsive_potential

        # Define the gradients of the cost with respect to both the decision variables
        df_z = 2 * self.gamma_r_lt * (ZZ - RR) - 2 * self.gamma_agg * (SS - ZZ) + 2 * self.gamma_rep*(SS - ZZ) + repulsive_gradient
        df_s = 2 * self.gamma_bar * (SS - b) + 2 * self.gamma_agg * (SS - ZZ) - 2 * self.gamma_rep*(SS - ZZ)

        return f_z, df_z, df_s

    def phi_agent(self, ZZ):
  
        '''This function performs an identity mapping on an input vector XX, returning
        the vector unchanged. Secondly, it facilitates the estimation of the aggregative variable
        (sigma) by providing an identity matrix of the same shape as the input vector.
        This latter is utilized in the calculation of the aggregative variable
        estimate, allowing each agent's decision variable to contribute to the estimation.
        '''

        return ZZ, np.eye(ZZ.shape[0])

    ######################################################################
    ###################### Aggregative Algorithm #########################
    ######################################################################

    def update_local_targets(self, RR, shift_x, shift_y, targets_motion_type):

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
            
            updated_RR[0] += shift_x  # Shift in the x-direction
            updated_RR[1] += 0  # Shift in the y-direction

        elif targets_motion_type == 'vert':

            updated_RR[0] += 0  # Shift in the x-direction
            updated_RR[1] += shift_y  # Shift in the y-direction

        elif targets_motion_type == 'diag':

            updated_RR[0] += shift_x  # Shift in the x-direction
            updated_RR[1] += shift_y  # Shift in the y-direction

        else:

            print("\nType of local targets'motion not allowed.\n")
            sys.exit()
            
        return updated_RR
    
    def aggregative_gradient_algorithm(self, ZZ, SS, VV, RR, FF, b):

        '''This function implements the aggregative optimization algorithm for a multi-robot system.
        It initializes decision variables, estimates of the aggregative variable, and cumulative
        gradients for each agent. Then, it iterates through optimization steps for a specified number
        of iterations. Within each iteration, it updates decision variables based on gradients and
        neighboring agents' information, calculates the estimate of the aggregative variable, and
        accumulates the cost function values. This process guides the collective movement of robots
        towards desired goals, such as the barycenter, while optimizing a defined cost function.'''

        #update the local targets
        if (self.kk % self.targets_rate_change == 0) and (self.moving_local_targets == True):
            
            print("Changing local targets!")

            #update local targets
            RR = self.update_local_targets(RR, self.shift_x, self.shift_y, self.targets_motion_type)

            #append the updated local targets to the list
            self.local_targets.append(RR.copy())

        #extract the message from each neighbor (to improve)
        data = np.zeros((13, 6 * self.n_z + 1))

        #extract the received message from each neighbor
        for jj in self.neigh: data[jj,:] = np.array(self.received_data[jj].pop(0)[1:]) 

        #compute the cost function and its gradient at the current decision variable
        _, df_z, df_s = self.cost_function(ZZ, SS, RR, b) 
        phi_k, dphi_k = self.phi_agent(ZZ)
        
        #update of the agent's decision variable using a steepest descent step 
        ZZ = ZZ - self.step_size * (df_z + dphi_k @ VV)
        phi_kp, _ = self.phi_agent(ZZ)

        #update of the estimate (tracker) of the aggregative variable sigma
        SS = self.WW[self.agent_id, self.agent_id] * SS + phi_kp - phi_k
        
        #sum the gradient's neighbors contribution
        for jj in self.neigh:
            SS_j = data[jj,2:4]
            SS += self.WW[self.agent_id, jj] * SS_j

        #compute cost and gradients for update
        f_i, df_z_plus, df_s_plus = self.cost_function(ZZ, SS, RR, b)  

        #update of the tracker of the gradient sum
        VV = self.WW[self.agent_id, self.agent_id] * VV + df_s_plus - df_s

        #sum the gradient's neighbors contribution
        for jj in self.neigh:
            VV_j = data[jj,4:6]
            VV += self.WW[self.agent_id, jj] * VV_j

        #store cost function and gradients
        FF = f_i
        self.grad_z = df_z_plus
        self.grad_s = df_s_plus

        return ZZ, SS, VV, FF, RR
    
    def timer_callback(self):

        '''
        This callback allows the agent to store its own message, i.e., its name,
        iteration and state, into the specific topic. In this way, all the neighbors
        will be able to read the message inside their own code in order to make the
        computation, reach the consensus, and hopefully evaluate the average value
        among the initial conditions, that turn out to be the best possible value.
        '''

        #if not self.first_iteration_done:
        if self.kk ==  0:
       
            #create, publish and show the message
            self.publish_message()

            #update iteration counter
            self.kk += 1

        #subsequent iterations
        else:

            #check if the list is either full or non/empty to proceed
            all_received = False
            
            #check whether all neighbors' messages have been received
            if all(len(self.received_data[j]) > 0 for j in self.neigh):

                all_received = all(self.kk-1  == self.received_data[j][0][0] for j in self.neigh)

            #check whether all the messages have been correctly received            
            if all_received:

                #perform the aggregative gradient tracking's iteration 
                self.ZZ, self.SS, self.VV, self.FF, self.RR = self.aggregative_gradient_algorithm(self.ZZ, self.SS, self.VV, self.RR, self.FF, self.b)
                
                #create, publish and show the message
                self.publish_message()

                #stop the node if kk exceeds the maximum iteration
                if self.kk > self.MAXITERS-1:

                    print("\n\n---------------------------------------------------------------")
                    print("End of Aggregative Tracking.")
                    print("---------------------------------------------------------------")
                    print("\nMAXITERS reached.\n")

                    #destroy the node
                    sleep(3)
                    self.destroy_node()
                
                #update iteration counter
                self.kk += 1

######################################################################
############################## Main ##################################
######################################################################

def main():

    #creates the node
    rclpy.init()

    #class initialization
    agent = Agent()

    #wait for synchronizattion
    agent.get_logger().info(f"Agent {agent.agent_id:d} -- Waiting for sync...")
    sleep(1)
    agent.get_logger().info("GO!")

    #let the node spin
    try:
        rclpy.spin(agent)

    except KeyboardInterrupt:
        agent.get_logger().info("----- Node stopped cleanly -----")

    finally:
        rclpy.shutdown() 
    
if __name__ == '__main__':
    main()
