import torch
from utils import hausdorff_distance_coef

# a = torch.tensor([
#     [0, 0, 0],
#     [1, 1, 1],
#     [0, 1, 0]
# ], dtype=torch.float16).unsqueeze(0).repeat(3, 1, 1, 1)
a = torch.randint(0, 2, (3, 1, 8, 8), dtype=torch.float16)
b = torch.randint(0, 2, (3, 1, 8, 8), dtype=torch.float16)
# print(a.squeeze(1).shape)

# b = torch.tensor([
#     [1, 1, 1],
#     [1, 0, 1],
#     [0, 1, 0]
# ], dtype=torch.float16).unsqueeze(0).repeat(3, 1, 1, 1)

hau = hausdorff_distance_coef(a, b)
print(hau)
