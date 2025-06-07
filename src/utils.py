import torch
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def convert_to_tensor(x, y, device):
    x = torch.FloatTensor(x).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    return x, y
