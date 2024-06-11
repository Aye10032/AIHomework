import torch
from utils import specificity_coef,sensitivity_coef

a = torch.tensor([
    [0, 0],
    [1, 1],
], dtype=torch.float16).unsqueeze(0).repeat(3, 1, 1, 1)
# a = torch.randint(0, 2, (3, 1, 8, 8), dtype=torch.float16)
# b = torch.randint(0, 2, (3, 1, 8, 8), dtype=torch.float16)
# print(a.squeeze(1).shape)

b = torch.tensor([
    [0, 1],
    [1, 0],
], dtype=torch.float16).unsqueeze(0).repeat(3, 1, 1, 1)

specificity = specificity_coef(a, b)
sensitivity = sensitivity_coef(a, b)
print(specificity, sensitivity)
