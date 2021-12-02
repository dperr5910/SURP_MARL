'''
File for the independent version of the environment
Ideally, the code would be housed in the same place
but that seems unlikely considering how the different policies interact
'''

from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from gym import spaces, Env
spaces.Discrete()



class Charging_Station_Env(Env):
    def __init__(self) -> None:
        super().__init_()
        self.reset()
        self.action_space = Box(0,self.required, (1,), float)
        self.observation_space = Box(0, 25, (1,), float)
        pass

    def reward(self):
        reward = 0
        reward -= (self.required**2)
        reward -= (self.load**2)
        
        return reward
        pass

    def done(self):
        if self.time >= 2:
            return True
        else:
            return False
        pass

    def step(self, action):
        self.time_step += 1
        self.load = 0
        self.load = action
        self.required -= self.load
        self.required = max(self.required, 0)

        observation = self.load
        reward = self.reward()
        done = self.done()
        
        return observation, reward, done, {}


    def reset(self):
        self.time_step = 0
        self.required = 2
        self.load = 0
        
        return 0



class Smart_Building_Environment(Env):
    def __init__(self) -> None:
        super().__init_()
        self.reset()
        self.action_space = Box(0,max(self.demand), (1,), float)
        self.observation_space = Box(0, 25, (1,), float)
        pass

    def reward(self):
        reward = 0
        reward -= abs(self.load - self.demands[self.time_step])**2
        reward -= (self.load**2)
        
        return reward
        pass

    def done(self):
        if self.time >= 2:
            return True
        else:
            return False
        pass

    def step(self, action):
        self.time_step += 1
        self.load = 0
        self.load = action
        observation = self.load
        reward = self.reward()
        done = self.done()
        
        return observation, reward, done, {}

    def reset(self):
        self.time_step = 0
        self.demands = [3,1,1]
        self.load = 0   
        return 0


class dual_environment(Env):
    def __init__(self) -> None:
        super().__init_()
        self.reset()
        self.action_space = Box(0,max(self.demand), (2,), float)
        self.observation_space = Box(0, 25, (2,), float)
        pass

    def reward(self):
        reward = 0
        reward -= abs(self.load - self.demands[self.time_step])**2
        reward -= (self.load**2)
        
        return reward
        pass

    def done(self):
        if self.time >= 2:
            return True
        else:
            return False
        pass

    def step(self, actions):
        self.time_step += 1
        self.load = 0
        self.load = actions[0] + actions[1]
        observation = self.load
        reward = self.reward()
        done = self.done()
        
        return observation, reward, done, {}

    def reset(self):
        self.time_step = 0
        self.demands = [3,1,1]
        self.load = 0   
        return 0