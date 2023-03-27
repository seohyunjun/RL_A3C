#import gym
import pickle
import  torch.multiprocessing as mp
from model import Model
from shared_optimizer import SharedAdam
from methods import train
import gymnasium as gym
# import numpy as np
# import random
import matplotlib.pyplot as plt

from utils import generate_gif

if __name__ == '__main__':
    #num_processes = mp.cpu_count()
    # seed = 1 
    num_processes = 6
    
    # np.random.seed(seed)
    # random.seed(seed)
    global_reward_score = []
    env_name = "CartPole-v1"
    env = gym.make(env_name, render_mode='rgb_array')
    shared_model = Model(env.observation_space.shape[0], env.action_space.n) # 4(), 2(left, right)  
    
    mp.set_start_method('spawn') 
    
    shared_model.share_memory()
    
    global_ep_no = mp.Value('i', 0) #init global episode no : 0
    
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=.0001, betas=(0.9,0.99), eps=1e-8, weight_decay=0.001)
    processes = []

    max_no_steps = 8000
    max_no_episodes = 5000
    
    for i in range(num_processes):
        p = mp.Process(target= train, args=(env_name, shared_model, max_no_steps, max_no_episodes, shared_optimizer, i, global_ep_no))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    
    
    with open(f'ep_total_reward_0.txt','rb') as f:
        ep_total_reward_0 = pickle.load(f)
        
    with open(f'ep_total_reward_1.txt','rb') as f:
        ep_total_reward_1 = pickle.load(f)
    
    with open(f'ep_total_reward_2.txt','rb') as f:
        ep_total_reward_2 = pickle.load(f)
    
    with open(f'ep_total_reward_3.txt','rb') as f:
        ep_total_reward_3 = pickle.load(f)
    
    with open(f'ep_total_reward_4.txt','rb') as f:
        ep_total_reward_4 = pickle.load(f)
    
    with open(f'ep_total_reward_5.txt','rb') as f:
        ep_total_reward_5 = pickle.load(f)
    

    plt.figure(figsize=(15,8))
    plt.scatter(range(len(ep_total_reward_0)),ep_total_reward_0, label = 'Episode 1', alpha=0.8)
    plt.scatter(range(len(ep_total_reward_1)),ep_total_reward_1, label = 'Episode 2', alpha=0.8)
    plt.scatter(range(len(ep_total_reward_2)),ep_total_reward_2, label = 'Episode 3', alpha=0.8)
    plt.scatter(range(len(ep_total_reward_3)),ep_total_reward_3, label = 'Episode 4', alpha=0.8)
    plt.scatter(range(len(ep_total_reward_4)),ep_total_reward_4, label = 'Episode 5', alpha=0.8)
    plt.scatter(range(len(ep_total_reward_5)),ep_total_reward_5, label = 'Episode 6', alpha=0.8)
    plt.legend()
    plt.savefig("global_reward_score_scatter.png")


    # generate_gif("episode_0/","episode_1")
    # generate_gif("episode_1/","episode_2")
    # generate_gif("episode_2/","episode_3")
    # generate_gif("episode_3/","episode_4")
    # generate_gif("episode_4/","episode_5")
    # generate_gif("episode_5/","episode_6")