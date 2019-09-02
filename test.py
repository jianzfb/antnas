# @Time    : 2019-08-23 15:28
# @Author  : zhangchenming
import torch.nn.functional as F
import torch

se_x = torch.randn([3, 2])
# print(se_x)
se_x = -se_x
print(se_x)
se_x = F.threshold(se_x, -1, -1)
print(se_x)
se_x = F.threshold(-se_x, 0, 0)

print(se_x)
