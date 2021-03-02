# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:45:26 2021

@author: user
"""
import numpy as np
import tensorflow as tf
import traci
import timeit
from enum import Enum

# self._Model = Model
# self._Memory = Memory

class LightPhase(Enum):
    PHASE_NS_GREEN = 0
    PHASE_NSL_GREEN = 2
    PHASE_EW_GREEN = 4
    PHASE_EWL_GREEN = 6
    
class Simulation:
    def __init__(self, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states):
        self._TrafficGen = TrafficGen
        # self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        # self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._total_wait_times = []
        self._avg_queue_length_store = []
        self._action_mapping = dict(zip([0, 1, 2, 3], [0, 2, 4, 6]))
        # self._training_epochs = training_epochs
    
    def run(self, ):
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=0)
        traci.start(self._sumo_cmd)
        print("Simulating...") #:>
        
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        self._waiting_times = {}
        old_state = -1
        old_action = -1
        
        
        while self._step < self._max_steps:
            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            
            action = self._choose_action(current_state, old_action, baseline=True)
            
            #we only trigger a yellow light if we aren't doing the same action again
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)  
            
            # print(action)
            self._set_green_phase(action)
            self._simulate(self._green_duration)
            
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait
            self._avg_queue_length_store.append(self._get_queue_length())
            self._total_wait_times.append(self._sum_waiting_time)
            
        
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 2)
        print("total wait time: ", self._sum_waiting_time)
        return simulation_time, self._avg_queue_length_store, self._total_wait_times
            
            
    def _choose_action(self, state, old_action, baseline=True):
        #simple periodic action sequence
        if baseline:
            action = (old_action + 1) % 4
            return action
    
    def _set_yellow_phase(self, old_action):
        yellow_phase_code = old_action * 2 +1
        traci.trafficlight.setPhase("TL", yellow_phase_code)
        
    def _set_green_phase(self, action):
        
        traci.trafficlight.setPhase("TL", self._action_mapping[action])
        
    def _simulate(self, steps_todo):
        steps_todo = min(steps_todo, self._max_steps - self._step)
        
        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length
            
            
    def _get_queue_length(self):
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _collect_waiting_times(self):
        incoming_roads = "E2TL N2TL W2TL S2TL".split()
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            if traci.vehicle.getRoadID(car_id) in incoming_roads:
                self._waiting_times[car_id] = traci.vehicle.getAccumulatedWaitingTime(car_id)
                # print(self._waiting_times[car_id])
            elif car_id in self._waiting_times:
                del self._waiting_times[car_id]
        tot_waiting_times = sum(self._waiting_times.values())
        print(tot_waiting_times)
        return tot_waiting_times
    
    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            # finding the lane where the car is located 
            # x2TL_3 are the "turn left only" lanes
            if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

            if valid_car:
                state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"

        return state

