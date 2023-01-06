import torch

class ParamScaler():

    def standardize(self, x):
        return x

    def original(self, x):
        return x

    def serialize(self):
        return {'name': 'ParamScaler'}


    @staticmethod
    def deserialize(d):
        return scalers[d['name']](*d['params'])


class MinMaxScaler(ParamScaler):

    def __init__(self, low, high):
        self.low = low
        self.range = high - self.low

    def standardize(self, x):
        return torch.clamp((x - self.low) / self.range, 0, 1)
    
    def original(self, x):
        return x * self.range + self.low

    def serialize(self):
        return {'name': 'MinMaxScaler', 'params': [float(self.low), float(self.range + self.low)]}

class MinCutoffScaler(ParamScaler):
    def __init__(self, low = 0):
        self.low = low

    def standardize(self, x):
        return torch.nn.functional.relu(x - self.low)

    def original(self, x):
        return x + self.low

    def to_dict(self):
        return {'name': 'MinCutoffScaler', 'params': [self.low]}

class NormalScaler(ParamScaler):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def standardize(self, x):
        return (x - self.mu) / self.sigma

    def original(self, x):
        return x * self.sigma + self.mu

    def serialize(self):
        return {'name': 'NormalScaler', 'params': [float(self.mu), float(self.sigma)]}

class HalfNormalScaler(NormalScaler):

    def standardize(self, x):
        return torch.nn.functional.relu(super().standardize(x))

    def serialize(self):
        return {'name': 'HalfNormalScaler', 'params': [float(self.mu), float(self.sigma)]}

scalers = {
    "ParamScaler": ParamScaler,
    "MinMaxScaler": MinMaxScaler,
    "MinCutoffScaler": MinCutoffScaler,
    "NormalScaler": NormalScaler,
    "HalfNormalScaler": HalfNormalScaler
}