import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import numpy as np
import time


#--- We now defined out Neural network we will use 
class DroneController(nn.Module):   #Create a blueprint for the brain of the drone
    def __init__(self, input_dim = 10, output_dim = 4):  #We have 10 inputs 3 for gyro, 3 for acceleration , 4 quaternian values
        super(DroneController, self).__init__()     #We are telling the upper neural network module to initialize this DorneCrontroller
        self.network = nn.Sequential(                  #This is setting up the strcuture for our strcuture for the neural netwrok.
            nn.Linear(input_dim, 32),              #We go from 10 - 32, 32-16, 16-4 
            nn.ReLU(),
            nn.Linear(32,16)
            nn.ReLU(),
            nn.Linear(16,output_dim)
            nn.Sigmoid()                             #Final function we use to finally get everything right
        )
    
    def forward(self, state):                       #Bascially saying the state or the values we get and push them through the neural network
        return self.network(state)
    

#----Mujoco Setup-----
model_path = "mujoco_menagerie/bitcraze_crazyfile_2/scene.xml"
try:
    model = mujoco.Mjmodel.from_xml_path(model_path)                       #Loading in details from the scene
    data = mujoco.MjData(model)

except ValueError as e:
    print(f"Error loading model: {e}")
    exit(1)

policy = DroneController()  #This is creating an instance of the drone controller we just made

#Sensor ID