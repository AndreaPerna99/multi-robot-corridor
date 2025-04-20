#Task 2: visualizer.py
#Gianluca Di Mauro, Andrea Perna, Meisam Tavakoli

######################################################################
######################### Libraries ##################################
######################################################################

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
import numpy as np
from std_msgs.msg import Float32MultiArray as msg_float

######################################################################
############################# Class ##################################
######################################################################

class Visualizer(Node):

    ######################################################################
    ######################### Initialization #############################
    ######################################################################

    def __init__(self):

        #allow parameters passing from the launch file
        super().__init__('visualizer', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        
        #get parameters from launcher
        self.agent_id = self.get_parameter('agent_id').value
        self.node_frequency = self.get_parameter('node_frequency').value

        #subscription to the topic to visualize
        self.subscription = self.create_subscription(msg_float, '/topic_{}'.format(self.agent_id),self.listener_callback, 10)

        #creates the timer for callback
        self.timer = self.create_timer(1.0/self.node_frequency, self.publish_data)
        
        #create the publisher that will communicate with Rviz
        self.publisher = self.create_publisher(Marker, '/visualization_topic', 1)

        #initialize message's contents
        self.current_pose = Pose()
        self.current_goal = Pose()

    ######################################################################
    ######################### Message Passing ############################
    ######################################################################

    def listener_callback(self, msg):

        '''The listener_callback function processes incoming messages received 
        on a subscribed topic within a ROS node. Specifically, it rearranges and
        stores data from the received message to update the current position of
        an agent. It fixes the x-coordinate of the agent's position based on a
        predetermined value (0.2*self.agent_id) and updates the y-coordinate using
        the third element (msg.data[2]) of the received message. This function
        essentially updates the current position of the agent based on the incoming
        data, facilitating real-time tracking and visualization of agent movement.'''

        #extract the message's content
        j = int(msg.data[0])
        msg_j = list(msg.data[1:])

        #extract the robot's position
        ZZ = np.array(msg_j[0:2])
        self.current_pose.position.x = ZZ[0]
        self.current_pose.position.y = ZZ[1]

        #extract the robot's goal
        RR = np.array(msg_j[6:8])

        self.current_goal.position.x = float(RR[0])
        self.current_goal.position.y = float(RR[1])

        #log message's content
        self.get_logger().info(f"\nAgent{self.agent_id} -> ZZ:({ZZ[0]},{ZZ[1]}), RR:({RR[0]},{RR[1]})\n")
        
    ######################################################################
    ######################### Rviz Visualization #########################
    ######################################################################

    def publish_data(self):

        '''The publish_data function encapsulates the process of preparing and sending
        visualization data to Rviz within a ROS node. It constructs a Marker message
        representing the current state of an agent, setting its position, type, color,
        and other properties based on received data. This message is then published to
        a topic using ROS publisher, enabling real-time visualization of agent movement
        or status in Rviz, aiding in robot monitoring and analysis within the ROS ecosystem.'''

        #set the agents' targets
        if (self.current_pose.position is not None) and (self.current_goal.position is not None):

            #set the type of message to send to Rviz
            agent_marker = Marker()

            #select the namespace
            agent_marker.header.frame_id = 'my_frame'
            agent_marker.header.stamp = self.get_clock().now().to_msg()

            #select the type of marker
            agent_marker.type = Marker.SPHERE

            #set the marker's pose
            agent_marker.pose.position.x = self.current_pose.position.x
            agent_marker.pose.position.y = self.current_pose.position.y
            agent_marker.pose.position.z = self.current_pose.position.z

            #select the marker's action
            agent_marker.action = Marker.ADD

            #select the namespace of the marker
            agent_marker.ns = 'agents'

            #set marker's ID
            agent_marker.id = self.agent_id

            #specify the scale of the marker
            scale = 0.2
            agent_marker.scale.x = scale
            agent_marker.scale.y = scale
            agent_marker.scale.z = scale

            #specify the color of the marker as RGBA
            color = [1.0, 0.0, 0.0, 1.0]
            if self.agent_id % 2:
                color = [0.0, 0.5, 0.5, 1.0]

            agent_marker.color.r = color[0]
            agent_marker.color.g = color[1]
            agent_marker.color.b = color[2]
            agent_marker.color.a = color[3]

            #publish the marker
            self.publisher.publish(agent_marker)

        #set the local targets' markers
        if self.current_goal.position is not None:

            #type of message
            marker = Marker()

            #select the namespace
            marker.header.frame_id = 'my_frame'
            marker.header.stamp = self.get_clock().now().to_msg()

            #select the type of marker
            marker.type = Marker.CUBE

            #set the marker's pose
            marker.pose.position.x = self.current_goal.position.x
            marker.pose.position.y = self.current_goal.position.y
            marker.pose.position.z = self.current_goal.position.z 

            #select the marker action
            marker.action = Marker.ADD

            #select the namespace of the marker
            marker.ns = 'loc_targets'

            #set marker's ID
            marker.id = self.agent_id

            #specify the scale of the marker
            scale = 0.2
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            #specify the marker's color
            color = [1.0, 0.0, 0.0, 1.0]
            if self.agent_id % 2:
                color = [0.0, 0.5, 0.5, 1.0]

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            #publish the marker
            self.publisher.publish(marker)

######################################################################
############################## Main ##################################
######################################################################

def main():

    rclpy.init()

    visualizer = Visualizer()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        print("----- Visualizer stopped cleanly -----")
    finally:
        rclpy.shutdown() 

if __name__ == '__main__':
    main()
