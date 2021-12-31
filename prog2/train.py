import numpy as np
import matplotlib.pyplot as plt
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
                    default=4000)
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
gamma = 0.99
env = gym.make('CartPole-v1')

print('Enviourment: CartPole-v1 \nNumber of actions: ' ,env.action_space.n,'\nDimension of state space: ',np.prod(env.observation_space.shape))
def run_episode(env, agent ,reward_to_go =False ,baseline=0., test_run=False):
    state = env.reset()
    rewards = []
    terminal = False
    episode_steps = 0
    # dW = np.zeros_like(agent.W)
    # db = np.zeros_like(agent.b)
    dw_actions, db_actions = [],[]
    
    while not terminal:
        episode_steps += 1
        action = agent.get_action(state)
        state, reward, terminal, _ = env.step(action)
        rewards.append(reward)
        #TODO fill in
        dw_action, db_action = agent.grad_log_prob(state,action)
        dw_actions.append(dw_action)
        db_actions.append(db_action)

    if(test_run):
        return None, None, sum(rewards)

    discount_factors = gamma ** np.array(range(1,episode_steps + 1))

    if(baseline):
        dw_baseline = np.multiply(np.square(dw_actions), np.array(rewards), axis=0).mean() / np.square(dw_actions).mean()
        db_baseline = np.multiply(np.square(db_actions), np.array(rewards)).mean() / np.square(db_actions).mean()
    else:
        dw_baseline = 0.0
        db_baseline = 0.0
    
    if(reward_to_go):
        for step in range(episode_steps):
            cummulated_reward_to_go =  np.array(rewards[step:]).T @ discount_factors[:episode_steps - step]
            dw_actions[step] *= (cummulated_reward_to_go - dw_baseline)
            db_actions[step] *= (cummulated_reward_to_go - db_baseline)
        dW = np.sum(np.array(dw_actions).reshape(episode_steps, *agent.W.shape), axis=0)
        db = np.sum(np.array(db_actions).reshape(episode_steps, *agent.b.shape), axis=0)
    else:
        discounted_cummulated_reward = discount_factors.T @ np.array(rewards)
        dW = np.sum(np.array(dw_actions).reshape(episode_steps, *agent.W.shape), axis=0) * (discounted_cummulated_reward - dw_baseline)
        db = np.sum(np.array(db_actions).reshape(episode_steps, *agent.b.shape), axis=0) * (discounted_cummulated_reward - db_baseline)

    return dW , db , sum(rewards)


def train(env, agent,args):
    rewards = []
    for i in range(args.N):
        dW = np.zeros_like(agent.W)
        db = np.zeros_like(agent.b)
        rewards.append(0)
        for j in range(args.b):
            #TODO fill in
            episode_dW, episode_db, episode_rewards = run_episode(env,agent,reward_to_go=True, baseline=1)
            dW += episode_dW / float(args.b)
            db += episode_db / float(args.b)
            rewards[i] += episode_rewards / float(args.b)
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
    print('Running 1000 test episodes....')
    for i in range(1000):
        _,_,r = run_episode(env,agent, test_run=True)
        rewards.append(r)
    rewards = np.array(rewards)
    rewards_to_plot = np.array([rewards[i - 200:i].mean() for i in range(200,1000)])
    print('Test reward {:.1f}{}{:.1f}'.format(np.mean(rewards),u"\u00B1",np.std(rewards)/np.sqrt(1000.)))
    return agent, rewards_to_plot
    # return agent, rewards


agent = PolicyGradientAgent(env, lr=args.lr)
agent, rewards = train(env,agent,args)
print('Average training rewards: ',np.mean(np.array(rewards)))
test(env,agent)
plt.plot(rewards)
plt.show()