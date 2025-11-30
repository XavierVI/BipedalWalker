import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --------------------------------------------------------------- #
#                     Policy Class Definitions                    #
# --------------------------------------------------------------- #
class SarsaQNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden, act):
        super().__init__()

        act_fn = getattr(F, act)
        self.layers = nn.ModuleList()
        prev = s_dim

        for h in hidden:
            self.layers.append(nn.Linear(prev, h))
            prev = h

        self.out = nn.Linear(prev, a_dim)
        self.act_fn = act_fn

    def forward(self, x):
        for l in self.layers:
            x = self.act_fn(l(x))

        return self.out(x)


class PolicyNet(nn.Module):
    def __init__(self, s_dim, a_dim, hidden, act):
        super().__init__()

        # this dynamically gets the activation function from torch.nn.functional
        act_fn = getattr(F, act)
        print(f"Using activation function: {act}")
        self.layers = nn.ModuleList()
        prev = s_dim
        
        for h in hidden:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        
        self.out = nn.Linear(prev, a_dim)
        self.act_fn = act_fn
    
    def forward(self, x):
        for l in self.layers:
            x = self.act_fn(l(x))

        x = self.out(x)

        # return logits and action probabilities
        return x, F.softmax(x, dim=-1)



# --------------------------------------------------------------- #
#                     Agent Class Definitions                     #
# --------------------------------------------------------------- #
class SarsaAgent:
    def __init__(self, cfg):
        # environment setup
        self.env = gym.make(cfg["env_name"], render_mode=None)
        self.render_env = gym.make(cfg["env_name"], render_mode="human")
        self.display = cfg.get("display", False)
        self.device = cfg.get("device", torch.device("cpu"))

        # state and action dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # hyperparameters
        self.gamma = cfg["gamma"]
        self.use_boltzmann = cfg.get("sarsa_use_boltzmann", False)

        # epsilon-greedy parameters
        self.epsilon = cfg["sarsa_initial_epsilon"]
        self.min_epsilon = cfg["sarsa_min_epsilon"]
        self.decay = cfg["sarsa_decay_rate"]
        
        # boltzmann parameters
        self.tau = cfg["sarsa_initial_tau"]
        self.min_tau = cfg["sarsa_min_tau"]
        self.decay_rate_tau = cfg["sarsa_decay_rate_tau"]

        self.render_int = cfg.get("display_episodes", 100)
        self.num_episodes = cfg["num_episodes"]
        self.max_steps = cfg.get("max_steps", 200)

        # policy network, optimizer, and loss function
        self.q_net = SarsaQNet(
            self.state_dim, self.action_dim,
            cfg["hidden_layers"], cfg["activation_function"]).to(self.device)

        self.opt = optim.Adam(self.q_net.parameters(), lr=cfg["learning_rate"])
        self.criterion = nn.MSELoss()

    def select_action(self, state: torch.Tensor):
        """
        Epsilon-greedy action selection.

        Args:
            state (np.array): Current state.
        Returns:
            action (int): Selected action.
        """
        # get Q-values from the network without tracking gradients
        # gradients are generated in the training loop
        with torch.no_grad():
            q_values = self.q_net(state)

        if self.use_boltzmann:
            prob_dist = F.softmax(q_values / self.tau, dim=-1)
            action = torch.multinomial(prob_dist, 1).item()

        else:
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)

            else:
                action = torch.argmax(q_values).item()

        return action

    def decay_epsilon(self):
        """
        Decay epsilon or tau.
        """
        if self.use_boltzmann:
            self.tau = max(
                self.min_tau, self.tau * np.exp(-self.decay_rate_tau))
        else:
            self.epsilon = max(
                self.min_epsilon, self.epsilon * np.exp(-self.decay))

    # Additional methods for training would go here
    def train(self):
        """
        SARSA training loop implementation.
        """
        print(f"Starting training with parameters gamma={self.gamma}, use_boltzmann={self.use_boltzmann}...")
        rewards_all = []

        for m in range(self.num_episodes):
            # total rewards for this episode
            total = 0

            env = self.render_env \
                if self.display and (m + 1) % self.render_int == 0 \
                else self.env
            
            # ==== EXPERIENCE COLLECTION PHASE ====
            experiences = []
            init_obs = env.reset()
            s = init_obs[0]
            a = self.select_action(torch.Tensor(s).to(self.device))

            # Gather experiences for the episode
            for t in range(self.max_steps):
                obs = env.step(a)
                s_next, r, done = obs[0], obs[1], obs[2]
                a_next = self.select_action(torch.Tensor(s_next).to(self.device))

                experiences.append((s, a, r, s_next, a_next, done))

                s = s_next
                a = a_next
                total += r
                
                if self.display and (m + 1) % self.render_int == 0:
                    env.render()

                if done:
                    break

            # ==== BATCH UPDATE PHASE ====
            # Convert experiences to batched tensors for parallel processing
            # First convert to numpy arrays, then to tensors (faster)
            states = torch.FloatTensor(
                np.array([exp[0] for exp in experiences])).to(self.device)
            actions = torch.LongTensor(
                np.array([exp[1] for exp in experiences])).to(self.device)
            rewards = torch.FloatTensor(
                np.array([exp[2] for exp in experiences])).to(self.device)
            next_states = torch.FloatTensor(
                np.array([exp[3] for exp in experiences])).to(self.device)
            next_actions = torch.LongTensor(
                np.array([exp[4] for exp in experiences])).to(self.device)
            dones = torch.FloatTensor(
                np.array([exp[5] for exp in experiences])).to(self.device)

            # Compute all Q-values in parallel (batched forward pass)
            q_values = self.q_net(states)
            # print("----- Q VALUES -----")
            # print(q_values.size())
            # Gather the predicted Q-values for the taken actions (actions.unsqueeze(1))
            pred_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # print("----- PREDICTED Q VALUES -----")
            # print(pred_q_values.size())

            # Compute target Q-values in parallel
            with torch.no_grad():
                next_q_values = self.q_net(next_states)
                next_q_for_actions = next_q_values.gather(
                    1, next_actions.unsqueeze(1)).squeeze(1)
                targets = rewards + self.gamma * next_q_for_actions * (1 - dones)

            # Single backward pass for entire episode
            loss = self.criterion(pred_q_values, targets)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            rewards_all.append(total)
            if self.use_boltzmann and (m + 1) % self.render_int == 0:
                print(
                    f"Episode {m+1}/{self.num_episodes} | Reward {total:.1f} | Avg {np.mean(rewards_all[-20:]):.1f} | τ={self.tau:.3f}", flush=True)
            elif (m + 1) % self.render_int == 0:
                print(
                    f"Episode {m+1}/{self.num_episodes} | Reward {total:.1f} | Avg {np.mean(rewards_all[-20:]):.1f} | ε={self.epsilon:.3f}", flush=True)

            self.decay_epsilon()

        return rewards_all


class ReinforceAgent:
    def __init__(self, cfg):
        self.env = gym.make(cfg["env_name"], render_mode=None)
        self.render_env = gym.make(cfg["env_name"], render_mode="human")
        self.display = cfg["display"]
        self.device = cfg["device"]
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        self.gamma = cfg["gamma"]
        self.use_boltzmann = cfg["reinforce_use_boltzmann"]

        # epsilon-greedy parameters
        self.epsilon = cfg["reinforce_initial_epsilon"]
        self.min_epsilon = cfg["reinforce_min_epsilon"]
        self.decay = cfg["reinforce_decay_rate"]
        
        # boltzmann parameters
        self.tau = cfg["reinforce_initial_tau"]
        self.min_tau = cfg["reinforce_min_tau"]
        self.decay_rate_tau = cfg["reinforce_decay_rate_tau"]

        self.render_int = cfg["display_episodes"]
        self.use_baseline = cfg["reinforce_use_baseline"]

        self.episodes = cfg["num_episodes"]
        self.max_steps = cfg["max_steps"]

        self.policy = PolicyNet(
            self.state_dim,
            self.action_dim,
            cfg["hidden_layers"],
            cfg["activation_function"]
        ).to(self.device)
        self.opt = optim.Adam(
            self.policy.parameters(),
            lr=cfg["learning_rate"]
        )

        if self.use_baseline:
            self.baseline = nn.Linear(self.state_dim, 1).to(self.device)
            self.baseline_opt = optim.Adam(self.baseline.parameters(), lr=cfg["learning_rate"])

    def select_action(self, state):
        # fetch the action probabilities from the policy network
        # (probability for each action)
        logits_t, probs_t = self.policy(
            torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )
        logits_t = logits_t.squeeze()
        probs_t = probs_t.squeeze()

        if self.use_boltzmann:
            # Numerically stable softmax with temperature
            scaled_logits = logits_t / self.tau
            boltz_probs = torch.softmax(scaled_logits, dim=-1)
            action = torch.multinomial(boltz_probs, 1).item()
            log_prob = torch.log(torch.clamp(boltz_probs[action], min=1e-8))
            
            return action, log_prob

        else:
            # Epsilon-greedy
            if np.random.rand() < self.epsilon:
                action = np.random.randint(self.action_dim)

            else:
                # create a categorical distribution over the actions
                dist = torch.distributions.Categorical(probs=probs_t)
                # sample an action from the distribution
                action = dist.sample()
                action = action.item()

        log_prob = torch.log(torch.clamp(probs_t[action], min=1e-8))

        return action, log_prob

    def update(self, log_probs, rewards, states):
        R, returns = 0, []

        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        if self.use_baseline:
            states_t = torch.FloatTensor(states).to(self.device)
            baselines = self.baseline(states_t).squeeze()
            advantages = returns - baselines.detach()
            base_loss = F.mse_loss(baselines, returns)
            self.baseline_opt.zero_grad()
            base_loss.backward()
            self.baseline_opt.step()
        else:
            advantages = returns
        
        # maximize expected return = minimize -expected return
        loss = -torch.sum(torch.stack([lp * adv for lp, adv in zip(log_probs, advantages)]))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def decay_epsilon(self):
        """
        Decay epsilon or tau.
        """
        if self.use_boltzmann:
            self.tau = max(
                self.min_tau, self.tau * np.exp(-self.decay_rate_tau))
        else:
            self.epsilon = max(
                self.min_epsilon, self.epsilon * np.exp(-self.decay))


    def train(self):
        print(f"Starting training with parameters gamma={self.gamma}, use_boltzmann={self.use_boltzmann}...")
        rewards_all = []

        for ep in range(self.episodes):
            env = self.render_env \
                if (ep + 1) % self.render_int == 0 and self.display \
                else self.env
            state, _ = env.reset()
            log_probs, rewards, states = [], [], []
            total = 0

            for t in range(self.max_steps):
                action, log_prob = self.select_action(state)

                log_probs.append(log_prob)
                
                next_state, reward, done, trunc, _ = env.step(action)
                
                rewards.append(reward)
                states.append(state)
                total += reward
                state = next_state

                if (ep + 1) % self.render_int == 0 and self.display:
                    env.render()

                if done or trunc:
                    break

            self.update(log_probs, rewards, states)
            self.decay_epsilon()
            rewards_all.append(total)
            
            if self.use_boltzmann and (ep + 1) % self.render_int == 0:
                print(
                    f"Episode {ep+1}/{self.episodes} | Reward {total:.1f} | Avg {np.mean(rewards_all[-20:]):.1f} | τ={self.tau:.3f}", flush=True)
            elif (ep + 1) % self.render_int == 0:
                print(
                    f"Episode {ep+1}/{self.episodes} | Reward {total:.1f} | Avg {np.mean(rewards_all[-20:]):.1f} | ε={self.epsilon:.3f}", flush=True)

        return rewards_all