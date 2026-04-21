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


#Sensor ID retrieval, we dont got names in mujoco, everything is ID based, so we need to retreive the ID via this method
gyro_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_gyro")  #This is getting the ID for the gyro sensor
acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_linacc")     #This is getting the ID for the acceleration sensor, Linacc stands for linear acceleration
quat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "body_quat")     #This is getting the ID for the quaternian sensor, which is basically the orientation of the drone



#--- Mujoco Simulation Loop ----
print("Starting simulation...")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()                          #All of this is bascially basic mujoco environment setup, easy peezy lemon sqeezy

        gyro = data.sensordata[model.sensor_adr[gyro_id]:model.sensor_adr[gyro_id]+3]  #This is retreiving the gyro data from the sensor data, we have to specify the range because we have 3 values for the gyro
        acc = data.sensordata[model.sensor_adr[acc_id] : model.sensor_adr[acc_id]+3]   #Same as above since we have 3 acceleration value
        quat = data.sensordata[model.sensor_adr[quat_id] : model.sensor_adr[quat_id]+4]   #Naa tho here we need 4 hence the +4

        #NOW lets make 1 big 10 element array combining 3 gyros, 3 accels, and 4 quats
        state = np.concatenate([gyro, acc, quat])   #This is combining all the sensor data into one big array that we can feed into our neural network
        state_tensor - torch.FloatTensor(state)   #Now we need to convert this numpy array into a torch tensor so we can feed it into our neural network.  Pytorch refuses to read numpy arrays.
        #BOOO PYTORCH, we demand respect for NUMPY people. LOL

        