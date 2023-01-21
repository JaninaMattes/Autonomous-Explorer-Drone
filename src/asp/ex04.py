import numpy as np; import torch as th; import scipy as sp; import gym 
import os; from collections import deque; import matplotlib.pyplot as plt

# RLLab Magic for calculating the discounted return G(t) = R(t) + gamma * R(t-1) 
# cf. https://github.com/rll/rllab/blob/ba78e4c16dc492982e648f117875b22af3965579/rllab/misc/special.py#L107
cumulate_discount = lambda x, gamma: sp.signal.lfilter([1], [1, - gamma], x[::-1], axis=0)[::-1]

class Net(th.nn.Module):
  def __init__(self, shape, activation, lr):
    super().__init__()
    self.net =  th.nn.Sequential(*[ layer 
      for io, a in zip(zip(shape[:-1], shape[1:]), [activation] * (len(shape)-2) + [th.nn.Identity] ) 
        for layer in [th.nn.Linear(*io), a()]])
    self.optimizer =  th.optim.Adam(self.net.parameters(), lr=lr)

class ValueNet(Net):
  def __init__(self, obs_dim, hidden_sizes=[64,64], activation=th.nn.Tanh, lr=1e-3):
    super().__init__([obs_dim] + hidden_sizes + [1], activation, lr)
  def forward(self, obs): return self.net(obs)
  def loss(self, states, returns): return ((returns - self(states))**2).mean()

class PolicyNet(Net):
  def __init__(self, obs_dim, act_dim, hidden_sizes=[64,64], activation=th.nn.Tanh, lr=3e-4):
    super().__init__([obs_dim] + hidden_sizes + [act_dim], activation, lr)
    self.distribution = lambda obs: th.distributions.Categorical(logits=self.net(obs))

  def forward(self, obs, act=None, det=False):
    """Given an observation: Returns policy distribution and probablilty for a given action 
      or Returns a sampled action and its corresponding probablilty"""
    pi = self.distribution(obs)
    if act is not None: return pi, pi.log_prob(act)
    act = self.net(obs).argmax() if det else pi.sample()
    return act, pi.log_prob(act)

  def loss(self, states, actions, advantages): 
    _, logp = self.forward(states, actions)
    loss = -(logp * advantages).mean()
    return loss

class PPO:
  """ Autonomous agent using vanilla policy gradient. """
  def __init__(self, env, seed=42,  gamma=0.99):
    self.env = env; 
    self.gamma = gamma;                       # Setup env and discount 
    th.manual_seed(seed);np.random.seed(seed);env.seed(seed)  # Seed Torch, numpy and gym
    # Keep track of previous rewards and performed steps to calcule the mean Return metric
    self._episode, self.ep_returns, self.num_steps = [], deque(maxlen=100), 0
    # Get observation and action shapes
    obs_dim = env.observation_space.shape[0] 
    act_dim = env.action_space.n                
    self.vf = ValueNet(obs_dim)             # Setup Value Network (Critic)
    self.pi = PolicyNet(obs_dim, act_dim)   # Setup Policy Network (Actor)

  def step(self, obs):
    """ Given an observation, get action and probs from policy and values from critc"""
    with th.no_grad(): 
      (a, prob), v = self.pi(obs), self.vf(obs)
    return a.numpy(), v.numpy()

  def policy(self, obs, det=True): return self.pi(th.tensor(obs), det=det)[0].numpy()

  def finish_episode(self):
    """Process self._episode & reset self.env, Returns (s,a,G,V)-Tuple and new inital state"""
    s, a, R, V = (np.array(e) for e in zip(*self._episode)) # Get trajectories from rollout
    self.ep_returns.append(sum(R))
    self._episode = []                                      # Add epoisode return to buffer & reset
    return (s,a,R,V), self.env.reset()                      # state, action, Return, Value Tensors + new state

  def collect_rollout(self, state, n_step=1):               # n_step=1 -> Temporal difference 
    rollout, done = [], False                               # Setup rollout buffer and env
    for _ in range(n_step):                                 # Repeat for n_steps 
      action, value = self.step(th.tensor(state))
      #act = action[0]           # Select action according to policy
      _state, reward, done, _ = self.env.step(action)       # Execute selected action
      self._episode.append((state, action, reward, value))  # Save experience to agent episode for logging
      rollout.append((state, action, reward, value))        # Integrate new experience into rollout
      state = _state; 
      self.num_steps += 1                                   # Update state & step
      if done: 
        _, state = self.finish_episode()             # Reset env if done 
    
    s,a,R,V = (np.array(e) for e in zip(*rollout))          # Get trajectories from rollout
    value = self.step(th.tensor(state))[1]                  # Get value of next state 
    A = G = cumulate_discount(R, self.gamma)                # REINFORCE Advantages (TODO 4-1)
    # A = G - V                                               # Actor Critic Advantages (TODO 4-1)
    A = R + self.gamma * np.append(V[1:], value) - V        # TD Actor-Critic Advantages (TODO 4-1)
    return (th.tensor(x.copy()) for x in (s,a,G,A)), state  # state, action, Return, Advantage Tensors 

  def train(self, states, actions, returns, advantages):        # Update policy weights
    self.pi.optimizer.zero_grad(); self.vf.optimizer.zero_grad()# Reset optimizer
    policy_loss = self.pi.loss(states, actions, advantages)     # Calculate Policy loss
    policy_loss.backward(); self.pi.optimizer.step()            # Apply Policy loss 
    value_loss = self.vf.loss(states, returns)                  # Calculate Value loss
    value_loss.backward(); self.vf.optimizer.step()             # Apply Value loss 
    print(f"Policy loss: {policy_loss}")
    print(f"Value loss: {value_loss}")

  def learn(self, steps):
    state, stats = self.env.reset(), []                         # Setup Stats and initial state
    while self.num_steps < steps:                               # Train for |steps| interatcions
      rollout, state = self.collect_rollout(state)              # Collect Rollout 
      stats.append((self.num_steps, np.mean(self.ep_returns)))  # Early Stopping and logging 
      if np.mean(self.ep_returns) >= self.env.spec.reward_threshold: return stats  
      print(f"At Step {self.num_steps:5d} Mean Return {stats[-1][1]:.2f}", end="\r", flush=True)
      self.train(*rollout)                                      # Perfom Update using rollout 
    return stats

if __name__ == '__main__':
  env = gym.make('CartPole-v1')       # Setup Env ['CartPole-v1'|'Acrobot-v1'|'MountainCar-v1']
  agent = PPO(env)                    # Setup PPO Agent 
  stats = agent.learn(steps=100000)   # Train Agent 

  # Find output directory & save final video + training progress 
  dir = f"./results/run_{len(next(os.walk('./results'))[1])}"
  env = gym.wrappers.Monitor(agent.env, dir, force=True)
  state, done = env.reset(), False
  while not done: 
    state,_,done,_ = env.step(agent.policy(state))
  plt.plot(*zip(*stats)); plt.title("Progress")
  plt.xlabel("Timestep"); plt.ylabel("Mean Return")
  plt.savefig(f"{dir}/training.png")
  