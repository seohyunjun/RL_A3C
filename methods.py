import os
import torch
import gym
from model import  Model
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
from utils import capture 
def calculate_target(final_v, rewards, gamma): # target value
    target = []
    for reward in reversed(rewards):
        final_v = reward + gamma * final_v
        target.append(final_v)
    target.reverse()
    return torch.as_tensor(target)

def train(env_name, shared_model, max_no_steps, max_no_episodes, shared_optimizer, process_id, global_ep_no):
    env = gym.make(env_name,render_mode="rgb_array")
    
    observation_no = env.observation_space.shape[0] # 4
    action_no = env.action_space.n #2ㅌ

    
    model = Model(observation_no, action_no)
    global_tot_rew_ls = []
    
    for i in range(max_no_episodes):
        model.load_state_dict(shared_model.state_dict())
        state, _ = env.reset()
        time_step = 0
        
        ep_tot_rew = 0
        ep_rew, ep_act, ep_state = [], [], []
        while True:
            time_step = time_step + 1
            action = model.action(torch.Tensor(state))
            
            n_state, rew, terminated, truncated, _ = env.step(action.item())
            done =  truncated or terminated 

            ep_tot_rew  = ep_tot_rew + rew
            
            
            if i >=max_no_episodes-10:
                screen = env.render()
                
                if f"episode_{process_id}" not in os.listdir():
                    os.mkdir(f"episode_{process_id}")
                
                if f"try_{i}" not in os.listdir(f"episode_{process_id}"):
                    os.mkdir(f"episode_{process_id}/try_{i}")
                
                file_name = f"episode_{process_id}/try_{i}/{time_step}"
                capture(screen, n_state, ep_tot_rew, terminated, truncated, action, done,time_step ,file_name)
            
            ep_rew.append(rew)
            ep_act.append(action.item(0))
            ep_state.append(state)
            state = n_state
            if time_step > max_no_steps or done:
                global_tot_rew_ls.append(ep_tot_rew)
                
                if done:
                    targets = calculate_target(0, ep_rew, gamma=0.9) # done = final_reward (0)
                else:
                    _, critic_value = model.forward(torch.Tensor(n_state))
                    targets = calculate_target(critic_value, ep_rew, gamma=0.9) # sequnce Q calculate 
                loss = model.calculate_loss(targets=targets,states=torch.Tensor(ep_state), actions=torch.Tensor(ep_act))
                shared_optimizer.zero_grad()
                loss.backward()
                for g_net, l_net in zip(shared_model.parameters(), model.parameters()): # update global network
                    g_net._grad = l_net.grad
                    
                shared_optimizer.step()
                
                with global_ep_no.get_lock(): # global_ep_no multiprocessing lock
                    global_ep_no.value += 1
                print("Global Episode No: {} Process ID: {} Episode No: {} Total Reward: {}".format(global_ep_no.value,process_id, i, ep_tot_rew))
                
                if i >=max_no_episodes-10:                    
                    os.rename(f"episode_{process_id}/try_{i}", f"episode_{process_id}/try_{i}_{int(ep_tot_rew)}")
                
                with open(f'ep_total_reward_{process_id}.txt','wb') as f:
                    pickle.dump(global_tot_rew_ls,f)
                
                break


