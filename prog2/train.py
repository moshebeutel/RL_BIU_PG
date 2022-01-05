import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import rand
from PGAgent import PolicyGradientAgent
import gym
import argparse
from datetime import datetime

parser = argparse.ArgumentParser('')
parser.add_argument('--lr',
                    help='learning rate',
                    type=float,
                    default=.0001)
parser.add_argument('--N',
                    help='Number of updates',
                    type=int,
                    default=10000)
parser.add_argument('--b',
                    help='batch size',
                    type=int,
                    default=1)
parser.add_argument('--seed',
                    help='random seed',
                    type=int,
                    default=1)
parser.add_argument('--RTG',help='reward to go', dest='RTG', action='store_true')
parser.add_argument('--baseline',help='use baseline', dest='baseline', action='store_true')


args = parser.parse_args()
np.random.seed(args.seed)
gamma = 1.0
env = gym.make('CartPole-v1')

print('Enviourment: CartPole-v1 \nNumber of actions: ' ,env.action_space.n,'\nDimension of state space: ',np.prod(env.observation_space.shape))
def run_episode(env, agent ,reward_to_go =False ,baseline=0, test_run=False):
    state = env.reset()
    rewards = []
    terminal = False
    episode_steps = 0
    dw_actions, db_actions = [],[]
    
    while not terminal:
        # if(test_run):
        #     env.render()
        episode_steps += 1
        action = agent.get_action(state)
        state, reward, terminal, _ = env.step(action)
        rewards.append(reward)
        if not test_run:
            dw_action, db_action = agent.grad_log_prob(state,action)
            dw_actions.append(dw_action)
            db_actions.append(db_action)

    if(test_run):
        return None, None, sum(rewards)

    discount_factors = gamma ** np.array(range(1,episode_steps + 1))

    assert not (np.isnan(dw_actions).any() or np.isnan(db_actions).any())   
    if(baseline):
        sqr_w = np.square(dw_actions)
        reward_like_w = np.array([np.ones_like(agent.W) * rewards[i] for i in range(episode_steps)]).reshape((episode_steps, *agent.W.shape))
        dw_baseline =np.mean(np.multiply(sqr_w, reward_like_w),axis=0) / np.mean(sqr_w, axis=0)
        dw_baseline[np.isnan(dw_baseline)] = 0.0
        sqr_b = np.square(db_actions)
        reward_like_b = np.array([np.ones_like(agent.b) * rewards[i] for i in range(episode_steps)]).reshape((episode_steps, *agent.b.shape))
        db_baseline = np.mean(np.multiply(sqr_b, reward_like_b),axis=0) / np.mean(sqr_b, axis=0)
        db_baseline[np.isnan(db_baseline)] = 0.0
    else:
        dw_baseline = 0.0
        db_baseline = 0.0
    
    if(reward_to_go):
        for step in range(episode_steps):
            cummulated_reward_to_go =  np.array(rewards[step:]).T @ discount_factors[:episode_steps - step]
            cummulated_reward_to_go_w = np.ones_like(dw_baseline) * cummulated_reward_to_go
            dw_actions[step] *= (cummulated_reward_to_go_w - dw_baseline)
            cummulated_reward_to_go_b = np.ones_like(db_baseline) * cummulated_reward_to_go
            db_actions[step] *= (cummulated_reward_to_go_b - db_baseline)
        dW = np.sum(np.array(dw_actions).reshape(episode_steps, *agent.W.shape), axis=0)
        db = np.sum(np.array(db_actions).reshape(episode_steps, *agent.b.shape), axis=0)
    else:
        discounted_cummulated_reward = discount_factors.T @ np.array(rewards)
        discounted_cummulated_reward_w =  np.ones_like(dw_baseline) * discounted_cummulated_reward
        dW = np.sum(np.array(dw_actions).reshape(episode_steps, *agent.W.shape), axis=0) * (discounted_cummulated_reward_w - dw_baseline)
        discounted_cummulated_reward_b =  np.ones_like(db_baseline) * discounted_cummulated_reward
        db = np.sum(np.array(db_actions).reshape(episode_steps, *agent.b.shape), axis=0) * (discounted_cummulated_reward_b - db_baseline)
    assert not (np.isnan(dW).any() or np.isnan(db).any())  
    return dW , db , sum(rewards)


def train(env, agent,N, b, rtg, baseline):
    rewards = []
    for i in range(N // b):
        dW = np.zeros_like(agent.W)
        db = np.zeros_like(agent.b)
        rewards.append(0)
        for j in range(b):
            #TODO fill in
            episode_dW, episode_db, episode_rewards = run_episode(env,agent,reward_to_go=rtg, baseline=baseline)
            dW += episode_dW / float(b)
            db += episode_db / float(b)
            rewards[i] += episode_rewards / float(b)
        agent.update_weights(dW, db)
        if i%100 == 25:
            temp = np.array(rewards[i - 25:i])
            dateTimeObj = datetime.now()
            timestampStr = dateTimeObj.strftime("%H:%M:%S")
            print('{}: [{}-{}] reward {:.1f}{}{:.1f}'.format(timestampStr,i-25,i,np.mean(temp),u"\u00B1",np.std(temp)/np.sqrt(25)))
    return agent, rewards

def test(env, agent):
    rewards = []
    print('_________________________')
    print('Running 500 test episodes....')
    for i in range(500):
        _,_,r = run_episode(env,agent, test_run=True)
        rewards.append(r)
    print('Test reward {:.1f}{}{:.1f}'.format(np.mean(rewards),u"\u00B1",np.std(rewards)/np.sqrt(500.)))
    return agent, rewards

for b in [1, 5, 10]:
    for rtg in [False, True]:
        for baseline in [0,1]:
            if not baseline or b > 1:
                for lr in [0.0001,0.001]:
                    agent = PolicyGradientAgent(env, lr=lr)
                    agent, rewards = train(env,agent,args.N,b,rtg,baseline)
                    print('Average training rewards: ',np.mean(rewards))
                    _,test_rewards = test(env,agent)
                    smoothed_train_rewards = np.array([np.mean(rewards[i - 20:i])for i in range(20,len(rewards))])
                    str = f'learning rate {lr} batch size {b} baseline {baseline} reward_to_go {rtg} '
                    plt.plot(smoothed_train_rewards)
                    plt.savefig(str + f'avg train reward  {np.mean(rewards)}' +  '.png')
                    plt.clf()
                    plt.plot(test_rewards)
                    plt.savefig(str + f'avg test reward  {np.mean(test_rewards)}' +  '.png')
                    plt.clf()

