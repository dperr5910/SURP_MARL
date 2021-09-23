import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario




class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        #Scenario Properties
        num_agents = 2
        num_adversaries = 0

        world.dim_c = 1
        world.dim_p = 2
        world.collaborative = True

        self.load = 0
        self.peak = 10
        self.energy_costs = []
        self.comfort = 0
        self.occupied = True
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
        

        world.landmarks = [Landmark() for i in range(3)]
        #Generate Landmarks
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

    def smart_building_reward(self, agent, world):
        reward = 0
        
        if self.occupied:
            reward += self.comfort
        
        reward -= self.load

        return reward

    def charging_station_reward(self, agent, world):
        if agent.required == 0:
            return 100
        return -(agent.rate*4+(agent.required))
        

    def observation(self, agent, world):
        print("LOAD")
        print(self.load)
        print("LOAD")
        for landmark in world.landmarks:
            if landmark.name == "Load":
                landmark.size = .1 * abs(self.load) + .1
                landmark.color = np.array([0.1+(self.load/self.peak/0.9), 1-(self.load/self.peak),0])
                #landmark.color = np.array([1,0,0])
            elif landmark.name == "comfort":
                landmark.size = self.comfort + .1

        if agent.name == "Charging_Station":
            self.load -= agent.rate
            agent.rate= agent.state.c[0]* 5
            if agent.required < 0:
                agent.required = 0
            elif agent.required > 0 and agent.rate > agent.required:
                agent.required = 0
            elif agent.required > 0 and agent.required > agent.rate:
                agent.required -= agent.rate
            self.load += agent.rate
            print("~~~~~~~~~")
            print(agent.state.c)
            print(agent.rate)
            print(agent.required)
            print("~~~~~~~~~")
            return([agent.rate, agent.required, self.load])
        elif agent.name  == "Smart_Building":
            print("energy")
            print(agent.energy)
            self.load -= agent.energy
            agent.energy = agent.state.c[0] * self.peak/2
            self.load += agent.energy
            self.comfort = agent.energy * agent.comfort
            return([agent.energy, agent.comfort, self.load])
        



        
    def smart_buildings(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Smart_Building")]

    def charging_stations(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Charging_Station")]

