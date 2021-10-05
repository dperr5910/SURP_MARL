import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random
import matplotlib.pyplot as plt
import pandas as pd



class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        #Scenario Properties
        num_agents = 2
        num_adversaries = 0
        world.dim_c = 1
        world.dim_p = 2
        world.collaborative = True
        self.data_path = r'C:\Users\perry\Documents\SURP_research\DEEPQ_SURP\PJME_hourly.csv'
        self.energy_costs = [1,1,1,2,2,3,4,5,6,7,7,7,10,10,10,10,9,8,8,8,4,3,3,2]
        #self.random_data()
        self.load = 0
        self.peak = 20
        self.comfort = 0
        self.occupied = True
        self.occupation = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0]
        self.done = False
        self.time = 0
        
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
        if agent.name == "Smart_Building":
            return self.smart_building_reward(agent, world)
        elif agent.name == "Charging_Station":
            return self.charging_station_reward(agent, world)
        
        return None
        pass



    def reset_world(self, world):
        self.load = 0
        self.comfort = 0
        self.done = False
        world.energy_costs = []

        
        #filling out agent detail
        for agent in world.agents:
            if agent.name == "Smart_Building":
                agent.energy = 0
                agent.comfort = 10 
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
                pass
            elif agent.name == "Charging_Station":
                agent.rate = 0
                agent.occupied = True
                agent.required = 72
                agent.state.p_pos = np.array([-.0,0.2])
                agent.color = np.array([0.8,0.5,0.8])
                agent.agent_callback = None 
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
        print("Observe - " + agent.name)
        print(self.load)
        for landmark in world.landmarks:
            if landmark.name == "Load":
                print(landmark.name)
                landmark.size = max(.1,min((1/self.peak * self.load),1))
                landmark.color = np.array([0.1+(self.load/self.peak/0.9), 1-(self.load/self.peak),0])
                #landmark.color = np.array([1,0,0])
            elif landmark.name == "comfort":
                landmark.size = .2 #self.comfort + .1

        if agent.name == "Charging_Station":
            self.states[0].append(agent.state.c)
            self.load -= agent.energy
            agent.rate= agent.state.c[0]* 5
            agent.energy = agent.rate * self.energy_cost[self.time]
            if agent.required < 0:
                agent.required = 0
            elif agent.required > 0 and agent.rate > agent.required:
                agent.required = 0
            elif agent.required > 0 and agent.required > agent.rate:
                agent.required -= agent.rate
            self.load += agent.energy
            return([agent.rate, agent.required, self.load])
        elif agent.name  == "Smart_Building":
            self.time += 1
            self.states[1].append(agent.state.c)
            self.load -= agent.energy
            agent.energy = agent.state.c[0] * self.energy_cost[self.time]
            self.load += agent.energy
            self.comfort = agent.energy * agent.comfort
            return([agent.energy, agent.comfort, self.load])

        
    def smart_building_reward(self, agent, world):
        reward = 0
        
        if self.occupied:
            reward += self.comfort
        
        reward -= agent.energy
        reward -= max(self.load, 0)
        return reward

    def charging_station_reward(self, agent, world):
        reward = 0
        if agent.required == 0:
            reward = 100 - agent.rate*10
        else:
            reward = -(agent.rate*2+(agent.required))
        return reward - max(self.load, 0)

    def random_data(self):
            df = pd.read_csv(self.data_path)
            df = df.sort_values('Datetime').reset_index(drop=True)
            index = random.randint(1,len(df))
            entry = df.iloc[index]
            while (not ("00:00:00" in entry["Datetime"])):
                index = random.randint(1,len(df))
                entry = df.iloc[index]
                pass
            self.energy_cost = []
            for count in range(index, index+24):
                self.energy_cost.append(float(df.iloc[count]["PJME_MW"]))

            amin, amax = min(self.energy_cost), max(self.energy_cost)
            for i, val in enumerate(self.energy_cost):
                self.energy_cost[i] = (val-amin) / (amax-amin)

            self.energy_cost = [ i + .5 for i in self.energy_cost]
            pass

        
    def smart_buildings(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Smart_Building")]

    def charging_stations(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Charging_Station")]

