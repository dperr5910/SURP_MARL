import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random
import matplotlib.pyplot as plt
import pandas as pd



class Scenario(BaseScenario):
    def __init__(self):
        self.day_reward = True
        self.method = "main"
        
    def make_world(self):
        world = World()

        #used for plotting actions done by policy. 
        #Future work: implementing it so that it can work with any number of input dimensions and entities
        world.actions = [[],[]]

        '''
        Scenario Properties
        '''

        num_agents = 2
        num_adversaries = 0
        world.dim_c = 1
        world.dim_p = 2

        if self.method == "main":
            world.collaborative = True
        elif self.method == "individual":
            world.collaborative = False


        self.energy_costs = [1,1,1,2,2,3,4,5,6,7,7,7,10,10,10,10,9,8,8,8,4,3,3,2,2]
        self.load = 0
        self.peak = 5
        self.comfort = 0
        self.occupied = True
        self.occupation = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
        self.done = False
        self.time = 0
        self.car_time = (8,16)
        self.day_reward = False
        
        #Generate Agents
        world.agents = [Agent() for i in range(2)]

        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.name = 'Smart_Building'
            elif i == 1:
                agent.name = "Charging_Station"
            agent.silent = False
            agent.movable = False
            agent.size = .1
        


        #Generate Landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.name = "Load"
                landmark.size = .2
            elif i==1:
                landmark.name = "energy cost"
                landmark.size = .1
            elif i ==2:
                landmark.name = "comfort"
                landmark.size = .1
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False

                
        self.reset_world(world)
        return world


    #reward method for calling. Deliberates to specific reward functions
    def reward(self, agent, world):
        self.day_reward= True
        reward = 0


        if agent.name == "Smart_Building":
            reward -= (agent.demands[self.time//2]-agent.energy)**2
            
            pass
            #reward = self.smart_building_reward(agent, world)
        elif agent.name == "Charging_Station":
            pass
            #reward = self.charging_station_reward(agent, world)

        reward -= self.load**2/2
        if (self.day_reward == False):
            return reward
        elif(self.day_reward == True and self.time == 6):
            print("reward")
            print(self.total_reward)
            tmp = self.total_reward
            self.total_reward =0
            return tmp
        else:
            print("d")
            self.total_reward += reward
            return 0
        print("not reward")
        return None



    def reset_world(self, world):
        self.load = 0
        self.time = 0
        self.comfort = 0
        self.done = False
        world.energy_costs = []
        self.total_reward = 0
        
        #filling out agent detail
        for agent in world.agents:
            if agent.name == "Smart_Building":
                agent.demands = [3,1,1]
                agent.min = 1
                agent.energy = 0
                #agent.comfort = 10 
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
                agent.max = 10
                agent.comfort_coef = .5
                pass
            elif agent.name == "Charging_Station":
                agent.rates = [0,self.peak]
                agent.prev_energy = 0
                agent.total = 0
                agent.energy = 0
                agent.rate = 0
                agent.occupied = True
                #charging deadline. after deadline, penalty is severe over more time
                agent.required = 72 #2
                agent.state.p_pos = np.array([-.0,0.2])
                agent.color = np.array([0.8,0.5,0.8])
                agent.agent_callback = None 
                agent.confidence = 0.5
                pass
            agent.state.c = np.zeros(world.dim_c)
            agent.action.c = np.zeros(world.dim_c)
            pass

        #filling out landmark detail
        for landmark in world.landmarks:
            if landmark.name == "Load":
                landmark.color = np.array([0.1+self.load/self.peak/0.9, 1-self.load/self.peak,0])
                landmark.state.p_pos = np.array([-.3,-.5])
                pass
            elif landmark.name == "energy cost":
                landmark.state.p_pos = np.array([0,-0.5])
                landmark.color = np.array([0.8,0.5,0.8])
                pass
            elif landmark.name == "comfort":
                landmark.state.p_pos = np.array([.2,-.5])
                landmark.color = np.array([0.1,0.5,0.8])
                pass

    '''
    Main observation method. Same as OpenAI observation.
    Obs - should be returning multiple 
    '''
    def observation(self, agent, world):
        self.time %= 6
        agent.prev_energy = agent.energy
        self.time += 1
        
        if (self.method != "main"):
            return self.rule(agent, world)
        for landmark in world.landmarks:
            if landmark.name == "Load":
                #print(landmark.name)
                landmark.size = max(.1,min((1/self.peak * self.load),1))
                landmark.color = np.array([0.1+(self.load/self.peak/0.9), 1-(self.load/self.peak),0])
                #landmark.color = np.array([1,0,0])
            elif landmark.name == "comfort":
                landmark.size = .2 #self.comfort + .1

        if agent.name == "Charging_Station":
            world.actions[0].append(agent.state.c)
            agent.energy= agent.state.c[0]* agent.rates[-1]
            agent.required -= agent.energy 
        elif agent.name  == "Smart_Building":
            world.actions[1].append(agent.state.c)
            agent.energy = agent.state.c[0] * agent.max

        self.load = self.new_load(world)
        return([agent.energy, self.load])



    def done(self,agent, world):
        if world.time == 48:
            return True
        else:
            return False
        

    #Reward functions used from 
    def smart_building_reward(self, agent, world):

        if self.occupation[self.time//2] == 1:
            reward = -self.cost(world)* max(agent.energy, 0) - agent.comfort_coef*((self.peak - agent.energy)**2)
        else:
            reward = -self.cost(world)* max(agent.energy, 0)

        reward -= abs(agent.energy-agent.prev_energy)*50
        if self.load>self.peak:
            reward -= 5000
        else:
            reward +=  -self.load * 100
        return reward

    def charging_station_reward(self, agent, world):

        reward = -agent.energy - agent.confidence*((agent.required - agent.total)**2)
        reward -= abs(agent.energy-agent.prev_energy)*50
        if self.load>self.peak:
            reward -= 5000
        else:
            reward += - self.load * 100
        if self.time//2 >= self.car_time[0] and self.time//2 <= self.car_time[1]:
            return reward
        else:
            return 0

        
    def smart_buildings(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Smart_Building")]

    def charging_stations(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Charging_Station")]
    
    '''
    Rule-based methods for environment.
    For future work, should either add custom policy or 

    max - maximum amount of power is distributed evenly among agents
    half - half of the max is distributed evenly among agents
    min - the minimum amount of energy is distributed to agents
    '''
    def rule(self, agent, world):
        if self.method == "max":
            self.no_sched(agent, world)
        elif self.method =="individual":
            self.individual(world)
        return [0,1]

    def no_sched(self,agent, world):
        if self.time == 0:
            if agent.name =="Charging_Station":
                agent.energy = agent.rates[-1]
            if agent.name =="Smart_Building":
                agent.energy = agent.demands[self.time//2]
        if self.time == 1:
            if agent.name =="Charging_Station":
                agent.energy = agent.rates[-1]
            if agent.name =="Smart_Building":
                agent.energy = agent.demands[self.time//2]
        if self.time == 2:
            if agent.name =="Charging_Station":
                agent.energy = 0
            if agent.name =="Smart_Building":
                agent.energy = agent.demands[self.time//2]
        self.load = 0
        for agent in world.agents:
            self.load += agent.energy
    pass



    def individual(self, world):
        for agent in world.agents:
            if self.time == 0:
                if agent.name =="Charging Station":
                    agent.energy = 0.66
                if agent.name =="Smart Building":
                    agent.energy = 1.5
            if self.time == 1:
                if agent.name =="Charging Station":
                    agent.energy = 0.66
                if agent.name =="Smart Building":
                    agent.energy = .5
            if self.time == 2:
                if agent.name =="Charging Station":
                    agent.energy = 0.66
                if agent.name =="Smart Building":
                    agent.energy = 0.5
        pass

    
    '''Gets the energy cost from multiple features attached to the world'''
    def cost(self, world):
        if self.load <= self.peak/3:
            cost = self.load
        elif self.load <= self.peak/3 * 2:
            cost = self.load**2
        else:
            cost = self.load**3
        return cost

    def new_load(self, world):
        load = 0
        for agent in world.agents:
            load += agent.energy
        return load
        


