# Simulated Self Driving Car

# Overview
This is the code for training a machine learning model to drive a simulated car using Convolutional Neural Networks on Udacity's self driving car simulator.

# Steps to run in autonomous mode:
1. Download Udacity's self driving car simulator from https://github.com/udacity/self-driving-car-sim.
2. Download Anaconda.
3. On anaconda prompt: conda env create -f environments.yml
4. On anaconda prompt: go to the path of this project using cd. e.g. cd C:\Users\Sougat\Documents\GitHub\Self-Driving-Car-Simulator\Self Driving Car Project
5. On anaconda prompt: conda info --envs (find your env name with path)
6. On anaconda prompt: conda activate <env name with path> e.g. conda activate C:\Users\Sougat\.conda\envs\car-behavioral-cloning
7. Update the model name in drive.py to the one which you want to use.
8. On anaconda prompt: python drive.py

# Steps to train model
1. Go to the training mode in the simulator
2. press 'r' to select the folder to store the images.
3. press 'r' again to start recording.
4. Press 'r' again to stop recording.
5. Do not move the IMG folder because your driving_log.csv contains the path to your images which shouldn't be changed.
6. Go to model.py and update the location of your driving_log.csv file.
7. On anaconda prompt: Go to the folder, activate the environment.
8. On anaconda prompt: python model.py

NOTE: Don't make changes to the github folder!  