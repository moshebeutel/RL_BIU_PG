import numpy as np
from numpy import random

eps = 0.00001

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class PolicyGradientAgent(object):
    def __init__(self, env, lr):
        self.num_actions = env.action_space.n
        self.num_features = env.observation_space.shape[0]
        self.W = np.random.rand(self.num_features, self.num_actions) * 0.01
        self.b = np.ones((self.num_actions)) * 0.01
        self.lr = lr
        self.last_probs = np.zeros((self.num_actions))

    def action_probability(self, state):
        '''
        Compute p(a|s) for discrete action using linear model on state
        :param state: environment state
        :return: vector of probabilities
        '''
        #TODO
        return softmax(self.W.T @ state + self.b)

    def get_action(self, state):
        '''
        Selects a random action according to p(a|s)
        :param state: environment state
        :return: action
        '''
        self.last_probs = self.action_probability(state)
        assert  1- eps < np.sum(self.last_probs) < 1 + eps
        assert self.last_probs.size == 2
        assert self.num_actions == 2
        self.last_probs[0] = 0.0 if self.last_probs[0] < eps else self.last_probs[0]
        self.last_probs[1] = 1.0 - self.last_probs[0]
        self.last_probs[1] = 0.0 if self.last_probs[1] < eps else self.last_probs[1]
        self.last_probs[0] = 1.0 - self.last_probs[1]
        r = random.rand()
        # return 0 if r < self.last_probs[0] else 1
        return np.random.choice(self.num_actions, p=self.last_probs)

    def grad_log_prob(self, state, action):
        '''
        Compute gradient of log P(a|S) w.r.t W and b
        :param state: environment state
        :param action: descrete action taken
        :return: dlogP(a|s)/dW, dlogP(a|s)/db
        '''
        # TODO
        # probs = self.last_probs.reshape(-1,1)   # shape (num_actions,1) (2,1)
        probs = self.action_probability(state).reshape(-1,1)
        dsoftmax = (np.diagflat(probs) - np.dot(probs, probs.T))[action, :]  
        assert probs[action,0] > eps
        dlogp_dz = (dsoftmax / probs[action,0]).reshape(1,-1)
        dz_dw = state.reshape(-1,1) # shape (num_features,)
        
        dW  = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dW[:,action] = (dz_dw @ dlogp_dz)[:,action]
        db[action] = dlogp_dz[:,action]  
        assert not (np.isnan(dW).any() or np.isnan(db).any())
        return  dW, db
        # return  (dz_dw @ dlogp_dz)[:,action], dlogp_dz[:,action]  


    def update_weights(self, dW, db):
        '''
        Updates weights using simple gradient ascent
        :param dW: gradients w.r.t W
        :param db: gradients w.r.t b
        '''
        assert not (np.isnan(dW).any() or np.isnan(db).any())
        self.W += self.lr * dW
        self.b += self.lr * db
        assert not (np.isnan(self.W).any() or np.isnan(self.b).any())
        # print(f'update_weights dW = {dW} db = {db} self.lr * dW {self.lr * dW} self.lr * db {self.lr * db}')


