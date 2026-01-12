# algorithms/ppo/rollout.py
class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []


    def add(self, obs, action, log_prob, reward, done):
        pass