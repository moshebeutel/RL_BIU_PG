import numpy as np

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
        probs = self.action_probability(state)
        return np.random.choice(self.num_actions, p=probs)

    def grad_log_prob(self, state, action):
        '''
        Compute gradient of log P(a|S) w.r.t W and b
        :param state: environment state
        :param action: descrete action taken
        :return: dlogP(a|s)/dW, dlogP(a|s)/db
        '''
        # TODO
        probs = self.action_probability(state).reshape(-1,1)   # shape (num_actions,1) (2,1)
        dsoftmax = (np.diagflat(probs) - np.dot(probs, probs.T))[action, :]  

        dlogp_dz = (dsoftmax / probs[action,0]).reshape(1,-1)
        dz_dw = state.reshape(-1,1) # shape (num_features,)
        
        dW  = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dW[:,action] = (dz_dw @ dlogp_dz)[:,action]
        db[action] = dlogp_dz[:,action]  
        return  dW, db
        # return  (dz_dw @ dlogp_dz)[:,action], dlogp_dz[:,action]  


    def update_weights(self, dW, db):
        '''
        Updates weights using simple gradient ascent
        :param dW: gradients w.r.t W
        :param db: gradients w.r.t b
        '''
        self.W -= self.lr * dW
        self.b -= self.lr * db

        # print(f'update_weights dW = {dW} db = {db} self.lr * dW {self.lr * dW} self.lr * db {self.lr * db}')


