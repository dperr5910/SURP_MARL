from stable_baselines3 import PPO
from environments import Smart_Building_Environment 
from environments import Charging_Station_Env 


def main():
    

    batch_size = 100
    ep_per_batch = 100


    SBE = Smart_Building_Environment()
    CSE = Charging_Station_Env()

    model_SBE = PPO('MlpPolicy', SBE, verbose = 0)
    model_CSE = PPO('MlpPolicy', CSE, verbose = 0)

    models = [model_SBE, model_CSE]

    for i in range(batch_size):
        for model in models:
            model.learn(total_timesteps=ep_per_batch)
        test(*models)



def test(model_SBE, model_CSE):
    pass
