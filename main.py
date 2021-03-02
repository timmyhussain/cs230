# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:53:49 2021

@author: user
"""
from simulation import Simulation
from generator import TrafficGenerator
from utils import import_train_configuration, set_sumo, set_test_path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    config = import_train_configuration(config_file="training_settings.ini")
    sumo_cmd = set_sumo(True, config['sumocfg_file_name'], config['max_steps'])
    
    TrafficGen = TrafficGenerator(config['max_steps'], config['n_cars_generated'])


    sim = Simulation(
        TrafficGen, 
        sumo_cmd, config['max_steps'],
        config['green_duration'],
        config['yellow_duration'], 
        config['num_states']
        )
    
    simulation_time, avg_queue_length, total_wait_times = sim.run()
    
 #%%
    
    plt.plot(range(len(avg_queue_length)), total_wait_times)
    plt.title("Total wait over time")
    plt.ylabel("Total delay /t")
    
    plt.figure()
    plt.plot(range(len(avg_queue_length)), avg_queue_length)
    plt.title("Average queue length over time")
    plt.ylabel("Number of cars in queue")