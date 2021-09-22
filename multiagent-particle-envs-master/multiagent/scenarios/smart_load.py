import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario




class Scenario(BaseScenario):
    def make_world(self):
        world = World()

        #Scenario Properties
        num_agents = 2
        num_adversaries = 0

        world.dim_c = 3
        #world.dim_p = None
        world.collaborative = True

        self.load = 0
        self.peak = 500
        self.energy_costs = []
        self.comfort = 0
        self.occupied = True
        
        #Generate Agents
        world.agents = [Agent() for i in range(2)]

        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.name = 'Smart_Building'
            elif i == 1:
                agent.name = "Charging_Station"
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
        world.energy_costs = []
        #filling out agent detail
        for agent in world.agents:
            if agent.name == "Smart_Building":
                agent.energy = 1
                agent.comfort = 10 
                agent.color = np.array([0.5,0.5,0.5])
                agent.state.p_pos = np.array([-.2,0.2])
                agent.agent_callback = None
                pass
            elif agent.name == "Charging_Station":
                agent.rate = 4
                agent.occupied = True
                agent.required = 72
                agent.state.p_pos = np.array([-.0,0.2])
                agent.color = np.array([0.8,0.5,0.8])
                agent.agent_callback = None 
                pass
            agent.state.c = np.zeros(world.dim_c)
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
        self.comfort = agent.state.c[1] * 50
        if self.occupied:
            reward += self.comfort
        self.load = 50 - agent.state.c[2]
        reward -= self.load

        return reward

    def charging_station_reward(self, agent, world):
        return 0
        

    def observation(self, agent, world):
        for landmark in world.landmarks:
            if landmark.name == "Load":
                landmark.color = np.array([0.1+self.load/self.peak/0.9, 1-self.load/self.peak,0])


        for agent in world.agents:
            if agent.name == "Charging_Station":
                return([agent.rate, agent.required, self.load])
            elif agent.name  == "Smart_Building":
                print(agent.state.c)
                self.load+= agent.energy
                return([agent.energy, agent.comfort, self.load])

        return [0,0,0]

    def smart_buildings(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Smart_Building")]

    def charging_stations(self, agent, world):
        return [agent for agent in world.agents if (agent.name == "Charging_Station")]

