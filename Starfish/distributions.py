import torch

class Normal(torch.distributions.Normal):
    def __init__(self, loc, scale, device = 'cpu'):
        super().__init__(torch.tensor(loc, dtype=torch.float64, device = device), scale=torch.tensor(scale, dtype = torch.float64, device = device))

class Uniform(torch.distributions.Uniform):
    def __init__(self, low, high, device = 'cpu'):
        super().__init__(torch.tensor(low, dtype=torch.float64, device = device), torch.tensor(high, dtype=torch.float64, device = device))

class HalfNormal(torch.distributions.HalfNormal):
    def __init__(self, scale, device = 'cpu'):
        super().__init__(torch.tensor(scale, dtype=torch.float64, device = device))