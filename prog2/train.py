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
                    default=.1)
parser.add_argument('--N',
                    help='Number of updates',
                    type=int,
                    default=2000)
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

env = gym.make('CartPole-v1')

print('Enviourment: CartPole-v1 \nNumber of actions: ' ,env.action_space.n,'\nDimension of state space: ',np.prod(env.observation_space.shape))
def run_episode(env, agent ,reward_to_go =False ,baseline=0.):
    state = env.reset()
    rewards = []
    terminal = False
    while not terminal:
        action = agent.get_action(state)
        state, reward, terminal, _ = env.step(action)
        rewards.append(reward)
        #TODO fill in
    return dW, db, sum(rewards)


def train(env, agent,args):
    rewards = []
    for i in range(args.N):
        dW = np.zeros_like(agent.W)
        db = np.zeros_like(agent.b)
        for j in range(args.b):
            #TODO fill in

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
        _,_,r,counter = run_episode(env,agent)
        rewards.append(r)
    rewards = np.array(rewards)
    print('Test reward {:.1f}{}{:.1f}'.format(np.mean(rewards),u"\u00B1",np.std(rewards)/np.sqrt(500.)))
    return agent, rewards


agent = PolicyGradientAgent(env)
agent, rewards = train(env,agent,args)
print('Average training rewards: ',np.mean(np.array(rewards)))
test(env,agent)
plt.plot(np.cumsum(rewards))
plt.show()