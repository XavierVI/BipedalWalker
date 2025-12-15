import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------- #
#                     Value Network Definitions                   #
# --------------------------------------------------------------- #
class ValueNet(nn.Module):
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


# --------------------------------------------------------------- #
#                     Policy Network Definitions                  #
# --------------------------------------------------------------- #
class PolicyNet(nn.Module):
    """Policy network for discrete action spaces."""

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


class ContinuousPolicyNet(nn.Module):
    """Policy network for continuous action spaces using Gaussian policy."""

    def __init__(self, s_dim, a_dim, hidden, act):
        super().__init__()

        act_fn = getattr(F, act)
        print(f"Using activation function: {act}")
        self.layers = nn.ModuleList()
        prev = s_dim

        for h in hidden:
            self.layers.append(nn.Linear(prev, h))
            prev = h

        # Output mean and log_std for Gaussian policy
        self.mean = nn.Linear(prev, a_dim)
        self.log_std = nn.Linear(prev, a_dim)
        self.act_fn = act_fn

    def forward(self, x):
        for l in self.layers:
            x = self.act_fn(l(x))

        mean = torch.tanh(self.mean(x))  # Bound mean to [-1, 1]
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Prevent numerical instability

        return mean, log_std
